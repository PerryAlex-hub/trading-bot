"""Venue ports.

One protocol, and deliberately read-only. There is no execution port here
because the bot has no execution: nothing in this package can place an order,
which is a structural guarantee rather than a configuration setting.

A Protocol rather than a base class so tests can substitute a plain stub
without inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Candle, ContractSpec, FundingInfo, Quote


@runtime_checkable
class MarketDataSource(Protocol):
    """Read-only venue access."""

    def klines(self, symbol: str, interval: str = "4h", limit: int = 200) -> list[Candle]:
        ...

    def quote(self, symbol: str) -> Quote | None:
        ...

    def funding(self, symbol: str) -> FundingInfo | None:
        ...

    def contract_specs(self, *, use_cache: bool = True) -> dict[str, ContractSpec]:
        ...


class VenueError(RuntimeError):
    """A venue rejected a request or behaved unexpectedly."""
