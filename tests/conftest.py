"""Shared fixtures.

Every test runs without a database server, without network and without
credentials. That is a hard requirement, not a convenience: CI has no secrets,
and a suite that needs them cannot run on a fork or catch the "works locally,
dies on deploy" class of bug.
"""

from __future__ import annotations

import datetime as dt
import os
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tradingbot.db.schema import Base
from tradingbot.models import Candle, ContractSpec


@pytest.fixture(autouse=True)
def isolate_settings_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the suite reading the developer's real .env file.

    Without this, `Settings()` in a test picks up whatever happens to be
    configured on the machine, so tests pass or fail depending on local
    credentials. That has already bitten this project twice: a bot test
    asserted a handler fails (true only while DATABASE_URL was unset) and an
    authorisation test asserted no chat ids exist (true only until a real
    TELEGRAM_CHAT_ID was added).

    Also clears the lru_cache on get_settings, so a cached instance built
    before isolation cannot leak into a test.
    """
    from tradingbot import settings as settings_module

    monkeypatch.setitem(settings_module.Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.startswith(
            ("TELEGRAM_", "DATABASE_", "WEEX_", "BINANCE_", "ANTHROPIC_", "TRADING_")
        ):
            monkeypatch.delenv(name, raising=False)
    settings_module.get_settings.cache_clear()
    yield
    settings_module.get_settings.cache_clear()


@pytest.fixture
def session() -> Session:
    """A fresh in-memory database per test."""
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def make_frame(
    *,
    n: int = 300,
    seed: int = 42,
    drift: float = 0.0,
    vol: float = 0.01,
    start: float = 100.0,
    freq: str = "4h",
) -> pd.DataFrame:
    """A reproducible OHLCV frame.

    Seeded so a failing assertion is reproducible. An unseeded random series
    makes a flaky test, and a flaky test in a trading system gets ignored.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, vol * 0.5, n)))
    low = close * (1 - np.abs(rng.normal(0, vol * 0.5, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = np.abs(rng.normal(1_000_000, 200_000, n))
    index = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_candles(
    *, n: int = 300, seed: int = 42, drift: float = 0.0, vol: float = 0.01,
    start: float = 100.0, interval_seconds: int = 14400,
) -> list[Candle]:
    """Candle objects ending on a recent closed-bar boundary."""
    frame = make_frame(n=n, seed=seed, drift=drift, vol=vol, start=start)
    now = int(dt.datetime.now(dt.UTC).timestamp())
    last_close = now - (now % interval_seconds) - interval_seconds
    candles: list[Candle] = []
    for offset, (_, row) in enumerate(frame.iterrows()):
        timestamp = last_close - interval_seconds * (n - 1 - offset)
        candles.append(
            Candle(
                open_time=dt.datetime.fromtimestamp(timestamp, tz=dt.UTC),
                open=Decimal(str(round(row["open"], 8))),
                high=Decimal(str(round(row["high"], 8))),
                low=Decimal(str(round(row["low"], 8))),
                close=Decimal(str(round(row["close"], 8))),
                volume=Decimal(str(round(row["volume"], 2))),
            )
        )
    return candles


def path_frame(path: list[float], *, wick: float = 0.4) -> pd.DataFrame:
    """Build a frame from an explicit price path.

    Used for pattern tests, where a deterministic shape is the whole point and
    a random walk would make the assertion meaningless.
    """
    close = np.asarray(path, dtype=float)
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close + wick,
            "low": close - wick,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=pd.date_range("2025-01-01", periods=len(close), freq="4h", tz="UTC"),
    )


def ramp(a: float, b: float, n: int) -> list[float]:
    return list(np.linspace(a, b, n))


@pytest.fixture
def btc_spec() -> ContractSpec:
    return ContractSpec(
        symbol="BTCUSDT",
        min_order_size=Decimal("0.0001"),
        size_step=Decimal("0.0001"),
        price_step=Decimal("0.1"),
        min_notional=Decimal("5"),
        max_leverage=125,
    )


@pytest.fixture
def goat_spec() -> ContractSpec:
    """A thin satellite contract: coarse step, same $5 floor.

    The interesting case for a small account, because the venue minimum binds
    long before the risk budget does.
    """
    return ContractSpec(
        symbol="GOATUSDT",
        min_order_size=Decimal("1"),
        size_step=Decimal("1"),
        price_step=Decimal("0.0001"),
        min_notional=Decimal("5"),
        max_leverage=20,
    )
