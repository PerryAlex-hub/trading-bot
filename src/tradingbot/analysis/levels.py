"""Support and resistance, volume profile, and the price levels a stop needs.

A level is only worth anything if it has been *tested* more than once. A single
pivot is a number; a price that rejected three separate approaches is a place
where orders actually sit. So levels here are built by clustering swing points
rather than by taking recent extremes, and every level carries its touch count
so the confluence layer can weight a 4-touch level above a 1-touch one.

Clustering tolerance is expressed in ATR, not percent. The universe spans BTC
at five figures and satellite tokens at four decimal places; a fixed percentage
tolerance that is sensible for one is absurd for the other.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .indicators import atr
from .structure import Swing, SwingType, confirmed_swings, find_swings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Level:
    """A clustered horizontal level."""

    price: float
    kind: Literal["support", "resistance"]
    touches: int
    first_touch: pd.Timestamp
    last_touch: pd.Timestamp
    # Spread of the cluster, in ATR. A tight cluster is a precise level; a wide
    # one is a zone, and a stop placed at its midpoint sits inside the noise.
    width_atr: float
    strength: float

    def distance_pct(self, price: float) -> float:
        if price <= 0:
            return float("inf")
        return abs(self.price - price) / price * 100.0

    def is_above(self, price: float) -> bool:
        return self.price > price


@dataclass(frozen=True)
class VolumeNode:
    """One bin of the volume profile."""

    price_low: float
    price_high: float
    volume: float

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2


@dataclass(frozen=True)
class VolumeProfile:
    """Where traded volume actually sits, by price.

    POC and value-area edges are better stop references than swing lows,
    because they mark prices the market agreed on rather than prices it merely
    touched once in a panic.
    """

    nodes: tuple[VolumeNode, ...]
    poc: float              # point of control: highest-volume price
    value_area_high: float
    value_area_low: float
    # Thin prices. Price tends to travel through them quickly, which makes them
    # poor targets and good places to expect slippage.
    low_volume_nodes: tuple[float, ...]

    def in_value_area(self, price: float) -> bool:
        return self.value_area_low <= price <= self.value_area_high


@dataclass(frozen=True)
class LevelMap:
    """Everything positional the strategy layer needs about price location."""

    supports: tuple[Level, ...]
    resistances: tuple[Level, ...]
    profile: VolumeProfile | None
    round_numbers: tuple[float, ...]

    def nearest_support(self, price: float) -> Level | None:
        below = [level for level in self.supports if level.price < price]
        return max(below, key=lambda level: level.price) if below else None

    def nearest_resistance(self, price: float) -> Level | None:
        above = [level for level in self.resistances if level.price > price]
        return min(above, key=lambda level: level.price) if above else None

    def headroom_ratio(self, price: float) -> float | None:
        """Distance to resistance divided by distance to support.

        The cheapest possible sanity check on a long: a ratio below 1 means
        there is more room to fall than to rise, which a trend-following signal
        on its own will never tell you.
        """
        support = self.nearest_support(price)
        resistance = self.nearest_resistance(price)
        if support is None or resistance is None:
            return None
        down = price - support.price
        if down <= 0:
            return None
        return (resistance.price - price) / down


def build_levels(
    frame: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
    tolerance_atr: float = 0.5,
    max_levels: int = 6,
    min_touches: int = 2,
    as_of: int | None = None,
) -> LevelMap:
    """Cluster confirmed swings into levels and build the volume profile."""
    if frame.empty:
        return LevelMap((), (), None, ())

    end = len(frame) - 1 if as_of is None else as_of
    window = frame.iloc[: end + 1]
    swings = confirmed_swings(find_swings(window, left=left, right=right), end)

    atr_value = _last_valid(atr(window, 14))
    if not atr_value or atr_value <= 0:
        # Fall back to a fraction of price so a very short series still yields
        # something usable rather than nothing.
        atr_value = float(window["close"].iloc[-1]) * 0.01

    tolerance = atr_value * tolerance_atr
    price = float(window["close"].iloc[-1])

    highs = [swing for swing in swings if swing.type is SwingType.HIGH]
    lows = [swing for swing in swings if swing.type is SwingType.LOW]

    resistances = _cluster(highs, tolerance, atr_value, "resistance", min_touches)
    supports = _cluster(lows, tolerance, atr_value, "support", min_touches)

    # A broken level flips role. Old resistance below price is now support, and
    # ignoring that would mean missing the most-watched levels on any chart.
    flipped_support = [
        Level(
            level.price, "support", level.touches, level.first_touch,
            level.last_touch, level.width_atr, level.strength,
        )
        for level in resistances
        if level.price < price
    ]
    flipped_resistance = [
        Level(
            level.price, "resistance", level.touches, level.first_touch,
            level.last_touch, level.width_atr, level.strength,
        )
        for level in supports
        if level.price > price
    ]

    all_supports = sorted(
        [level for level in supports if level.price < price] + flipped_support,
        key=lambda level: level.strength,
        reverse=True,
    )[:max_levels]
    all_resistances = sorted(
        [level for level in resistances if level.price > price] + flipped_resistance,
        key=lambda level: level.strength,
        reverse=True,
    )[:max_levels]

    return LevelMap(
        supports=tuple(sorted(all_supports, key=lambda level: level.price, reverse=True)),
        resistances=tuple(sorted(all_resistances, key=lambda level: level.price)),
        profile=build_volume_profile(window),
        round_numbers=_round_numbers(price),
    )


def _cluster(
    swings: Sequence[Swing],
    tolerance: float,
    atr_value: float,
    kind: Literal["support", "resistance"],
    min_touches: int,
) -> list[Level]:
    """Group nearby swings into levels.

    Single-linkage on sorted prices: walk upward and start a new cluster
    whenever the gap to the previous price exceeds the tolerance. Simple, and
    deterministic in a way k-means would not be - which matters, because a
    level map that shifts between runs on identical data makes the journal
    impossible to reason about.
    """
    if not swings:
        return []

    ordered = sorted(swings, key=lambda swing: swing.price)
    clusters: list[list[Swing]] = [[ordered[0]]]
    for swing in ordered[1:]:
        if swing.price - clusters[-1][-1].price <= tolerance:
            clusters[-1].append(swing)
        else:
            clusters.append([swing])

    levels: list[Level] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = [swing.price for swing in cluster]
        times = sorted(swing.time for swing in cluster)
        # Weight by pivot strength: a level built from three major pivots is
        # not the same as one built from three minor wiggles.
        weights = np.array([swing.strength + 1 for swing in cluster], dtype=float)
        centre = float(np.average(prices, weights=weights))
        width_atr = (max(prices) - min(prices)) / atr_value if atr_value else 0.0

        levels.append(
            Level(
                price=centre,
                kind=kind,
                touches=len(cluster),
                first_touch=times[0],
                last_touch=times[-1],
                width_atr=width_atr,
                strength=_level_strength(cluster, width_atr),
            )
        )
    return levels


def _level_strength(cluster: Sequence[Swing], width_atr: float) -> float:
    """Score a level: more touches, stronger pivots, tighter cluster.

    Recency is deliberately *not* a factor. A level from six months ago that
    has been respected four times is more meaningful than last week's single
    pivot, and most retail level-finders get this backwards.
    """
    touches = len(cluster)
    avg_strength = float(np.mean([swing.strength for swing in cluster]))
    tightness = 1.0 / (1.0 + width_atr)
    return touches * (1.0 + avg_strength / 10.0) * tightness


def build_volume_profile(
    frame: pd.DataFrame, bins: int = 40, value_area_pct: float = 0.70
) -> VolumeProfile | None:
    """Approximate a volume profile from OHLCV bars.

    Honest about its limits: without tick data, each bar's volume is spread
    uniformly across its high-low range. That smears a bar that actually traded
    almost all its size at one price. The POC it produces is still far more
    informative than nothing, but it is an approximation, not an order-book
    reconstruction.
    """
    if frame.empty or len(frame) < bins // 2:
        return None

    low = float(frame["low"].min())
    high = float(frame["high"].max())
    if high <= low:
        return None

    edges = np.linspace(low, high, bins + 1)
    volumes = np.zeros(bins)

    for bar_low, bar_high, bar_volume in zip(
        frame["low"].to_numpy(dtype=float),
        frame["high"].to_numpy(dtype=float),
        frame["volume"].to_numpy(dtype=float),
        strict=True,  # same frame, so equal length is an invariant worth asserting
    ):
        if bar_high <= bar_low or bar_volume <= 0:
            continue
        start = int(np.searchsorted(edges, bar_low, side="right") - 1)
        stop = int(np.searchsorted(edges, bar_high, side="left"))
        start = max(0, min(start, bins - 1))
        stop = max(start + 1, min(stop, bins))
        # Distribute proportionally to how much of each bin the bar covers.
        span = bar_high - bar_low
        for index in range(start, stop):
            overlap = min(bar_high, edges[index + 1]) - max(bar_low, edges[index])
            if overlap > 0:
                volumes[index] += bar_volume * (overlap / span)

    if volumes.sum() <= 0:
        return None

    nodes = tuple(
        VolumeNode(float(edges[i]), float(edges[i + 1]), float(volumes[i]))
        for i in range(bins)
    )
    poc_index = int(np.argmax(volumes))
    poc = float((edges[poc_index] + edges[poc_index + 1]) / 2)

    low_edge, high_edge = _value_area(volumes, edges, poc_index, value_area_pct)
    median = float(np.median(volumes[volumes > 0])) if (volumes > 0).any() else 0.0
    thin = tuple(
        node.mid for node in nodes if 0 < node.volume < median * 0.3
    )

    return VolumeProfile(
        nodes=nodes,
        poc=poc,
        value_area_high=high_edge,
        value_area_low=low_edge,
        low_volume_nodes=thin,
    )


def _value_area(
    volumes: np.ndarray, edges: np.ndarray, poc_index: int, target_pct: float
) -> tuple[float, float]:
    """Expand outward from the POC until `target_pct` of volume is enclosed.

    The standard construction: at each step take whichever adjacent bin holds
    more volume, so the area grows toward where trading actually happened
    rather than symmetrically.
    """
    total = volumes.sum()
    target = total * target_pct
    low_index = high_index = poc_index
    accumulated = volumes[poc_index]

    while accumulated < target and (low_index > 0 or high_index < len(volumes) - 1):
        below = volumes[low_index - 1] if low_index > 0 else -1.0
        above = volumes[high_index + 1] if high_index < len(volumes) - 1 else -1.0
        if above >= below:
            high_index += 1
            accumulated += above
        else:
            low_index -= 1
            accumulated += below

    return float(edges[low_index]), float(edges[high_index + 1])


def _round_numbers(price: float, count: int = 3) -> tuple[float, ...]:
    """Psychological round levels near the current price.

    Real clustering happens at round numbers because that is where humans and
    naive bots place orders. The step is derived from the price's own magnitude
    so it works for BTC at 90,000 and a token at 0.00042 alike.
    """
    if price <= 0:
        return ()
    magnitude = 10 ** np.floor(np.log10(price))
    step = float(magnitude / 2)
    base = float(np.floor(price / step) * step)
    # Cast to built-in float: numpy scalars are not JSON-serialisable, and
    # these values land in the `features` column of the journal.
    levels = {
        float(round(base + offset * step, 12))
        for offset in range(-count, count + 1)
    }
    return tuple(sorted(level for level in levels if level > 0))


def _last_valid(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    return float(cleaned.iloc[-1]) if not cleaned.empty else None
