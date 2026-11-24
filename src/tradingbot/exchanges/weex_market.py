"""WEEX public market data: quotes and funding rates.

**No authentication anywhere in this module, by design.** The bot does not place
orders, so it never needs an API key, a secret or a passphrase. That removes
the entire class of risk that comes with holding exchange credentials in a
cloud environment, and it removes the inferred-signing-scheme problem that the
earlier private client had to be defended against.

Why WEEX at all when the candles come from Binance: the risk gate's friction
checks need the spread and funding rate of the venue a trade would actually be
taken on. A 1:2 setup that loses a third of its edge to a wide satellite spread
is not a 1:2 setup, and Binance's spread does not tell you anything about
WEEX's book.

Everything here returns `None` on failure rather than raising. The gate decides
what missing data means; a venue hiccup must not crash a scheduled run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from ..models import FundingInfo, Quote
from .weex import WeexClient, WeexEndpointError, _first, _to_decimal, _unwrap

log = logging.getLogger(__name__)

# Paths from WEEX's published futures REST documentation. The `/capi/` entries
# are retained as fallbacks but are known to 404: they came from search results
# rather than the docs, which is why every deployed run could not fetch a quote.
TICKER_PATHS = (
    "/api/swap/v3/market/ticker",
    "/capi/v3/market/ticker",
)
ALL_TICKERS_PATHS = (
    "/api/swap/v3/market/tickers",
    "/capi/v3/market/tickers",
)
DEPTH_PATHS = ("/api/swap/v3/market/depth", "/capi/v3/market/depth")
FUNDING_PATHS = (
    "/api/swap/v3/market/funding_time",
    "/api/swap/v3/market/funding-rate",
    "/capi/v3/market/funding-rate",
)

# Documented ticker fields are snake_case: best_bid, best_ask, last. The
# camelCase variants are kept because third-party SDKs report them.
_BID_KEYS = ("best_bid", "bestBid", "bid", "buyPrice", "bidPx")
_ASK_KEYS = ("best_ask", "bestAsk", "ask", "sellPrice", "askPx")
_MARK_KEYS = ("last", "markPrice", "mark", "lastPrice", "close", "indexPrice")
_FUNDING_KEYS = ("funding_rate", "fundingRate", "rate", "currentFundingRate")


@dataclass
class WeexMarketData(WeexClient):
    """Public quotes and funding, on top of the contract-spec client.

    Subclasses `WeexClient` so the endpoint probing and field normalisation it
    already does are shared rather than duplicated. WEEX has moved its
    documented paths across /capi/v1, v2 and v3, and published field names
    disagree between sources, so probing is the only robust approach.
    """

    def quote(self, symbol: str) -> Quote | None:
        """Top of book plus mark price.

        Falls back to the order-book endpoint when the ticker does not report
        both sides. Returning a quote with a fabricated zero spread would be
        worse than returning nothing: the gate would approve a trade whose
        friction it never actually checked.
        """
        try:
            _path, payload = self._try_paths(TICKER_PATHS, params={"symbol": symbol})
        except WeexEndpointError as exc:
            log.warning("No WEEX ticker endpoint responded for %s: %s", symbol, exc)
            return None

        entry = self._pick_symbol_entry(payload, symbol)
        if entry is None:
            return self._quote_from_depth(symbol, None)

        bid = _to_decimal(_first(entry, _BID_KEYS))
        ask = _to_decimal(_first(entry, _ASK_KEYS))
        mark = _to_decimal(_first(entry, _MARK_KEYS))

        if bid is None or ask is None:
            return self._quote_from_depth(symbol, mark)
        if mark is None:
            mark = (bid + ask) / 2
        if bid <= 0 or ask <= 0 or ask < bid:
            log.warning("Nonsensical WEEX quote for %s: bid=%s ask=%s", symbol, bid, ask)
            return None

        return Quote(
            symbol=symbol, bid=bid, ask=ask, mark=mark, at=datetime.now(tz=UTC)
        )

    @staticmethod
    def _pick_symbol_entry(payload: object, symbol: str) -> dict | None:
        """Find our symbol's row.

        Some WEEX routes return one object and some return every contract, so
        taking the first entry of a list would silently read another symbol's
        prices - which would be far worse than failing.
        """
        entries = _unwrap(payload)
        if entries:
            wanted = symbol.upper().replace("-", "").replace("_", "")
            for entry in entries:
                raw = str(_first(entry, ("symbol", "contract_code", "contractCode")) or "")
                if raw.upper().replace("-", "").replace("_", "") == wanted:
                    return entry
            # A single-row response legitimately has no symbol field.
            return entries[0] if len(entries) == 1 else None
        return payload if isinstance(payload, dict) else None

    def _quote_from_depth(self, symbol: str, mark: Decimal | None) -> Quote | None:
        try:
            _path, payload = self._try_paths(
                DEPTH_PATHS, params={"symbol": symbol, "limit": 5}
            )
        except WeexEndpointError:
            return None

        container = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(container, dict):
            return None
        bids, asks = container.get("bids"), container.get("asks")
        if not bids or not asks:
            return None
        try:
            bid = Decimal(str(bids[0][0]))
            ask = Decimal(str(asks[0][0]))
        except (IndexError, TypeError, ValueError):
            return None
        if bid <= 0 or ask < bid:
            return None
        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            mark=mark or (bid + ask) / 2,
            at=datetime.now(tz=UTC),
        )

    def all_marks(self) -> dict[str, Decimal]:
        """Mark price for every contract, in one request.

        Exists so minimum order values can be ranked across the whole venue
        without one request per symbol. The minimum order size WEEX publishes
        is denominated in the BASE asset, so its cost in USDT is
        `min_order_size x price` - which means the cheapest contract to test
        with is a question about price, and cannot be answered from the spec
        list alone.
        """
        try:
            _path, payload = self._try_paths(ALL_TICKERS_PATHS)
        except WeexEndpointError as exc:
            log.warning("No WEEX all-tickers endpoint responded: %s", exc)
            return {}

        marks: dict[str, Decimal] = {}
        for entry in _unwrap(payload):
            raw = _first(entry, ("symbol", "contract_code", "contractCode"))
            if raw is None:
                continue
            mark = _to_decimal(_first(entry, _MARK_KEYS))
            if mark is None or mark <= 0:
                continue
            marks[str(raw).upper().replace("-", "").replace("_", "")] = mark
        return marks

    def funding(self, symbol: str) -> FundingInfo | None:
        """Current funding rate, normalised to a percentage per 8h.

        Venues report this inconsistently as a fraction (0.0001) or already as
        a percent (0.01). A value under 0.01 in absolute terms is almost
        certainly a fraction, because a 0.01% funding rate is unremarkable
        while a 1% one would be extraordinary.
        """
        try:
            _path, payload = self._try_paths(FUNDING_PATHS, params={"symbol": symbol})
        except WeexEndpointError:
            log.warning("No WEEX funding endpoint responded for %s", symbol)
            return None

        entry = self._pick_symbol_entry(payload, symbol)
        if entry is None:
            return None

        raw_rate = _to_decimal(_first(entry, _FUNDING_KEYS))
        if raw_rate is None:
            return None
        rate_pct = raw_rate * Decimal("100") if abs(raw_rate) < Decimal("0.01") else raw_rate
        return FundingInfo(symbol=symbol, rate_8h_pct=rate_pct)


def fetch_market_data(symbol: str) -> tuple[Quote | None, FundingInfo | None]:
    """Convenience helper for one symbol. Never raises."""
    try:
        with WeexMarketData() as client:
            return client.quote(symbol), client.funding(symbol)
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("Could not fetch WEEX market data for %s: %s", symbol, exc)
        return None, None
