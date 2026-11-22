"""Venue ports.

Two protocols, kept separate because they have very different risk profiles:
`MarketDataSource` is read-only and safe to call anywhere, while
`ExecutionVenue` moves money. Splitting them means the analysis layer can be
handed a market-data object that is structurally incapable of placing an order.

Protocols rather than base classes so the paper broker and the WEEX client are
interchangeable without inheritance, and so tests can substitute a plain stub.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, Sequence, runtime_checkable

from ..models import (
    Candle,
    ContractSpec,
    FundingInfo,
    OrderRequest,
    OrderResult,
    Quote,
    Side,
)


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


@runtime_checkable
class ExecutionVenue(Protocol):
    """Anything that can place and manage orders.

    Implemented by both the live WEEX client and the paper broker. The pipeline
    only ever talks to this interface, which is what allows paper and live to
    share one code path - and therefore what makes the paper results actually
    predictive of live behaviour.
    """

    @property
    def is_paper(self) -> bool:
        ...

    def place_order(self, request: OrderRequest) -> OrderResult:
        ...

    def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        ...

    def open_positions(self) -> list[VenuePosition]:
        ...

    def account_equity(self) -> Decimal | None:
        ...


class VenuePosition:
    """A position as the venue reports it.

    Separate from the database `Position` row on purpose. Reconciliation
    compares the two, and sharing one type would make it impossible to
    represent the disagreement that reconciliation exists to detect.
    """

    __slots__ = ("symbol", "side", "quantity", "entry_price", "unrealised_pnl", "leverage", "raw")

    def __init__(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        entry_price: Decimal,
        unrealised_pnl: Decimal | None = None,
        leverage: Decimal | None = None,
        raw: dict | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.unrealised_pnl = unrealised_pnl
        self.leverage = leverage
        self.raw = raw or {}

    def __repr__(self) -> str:
        return (
            f"VenuePosition({self.symbol} {self.side.value} "
            f"{self.quantity} @ {self.entry_price})"
        )


class VenueError(RuntimeError):
    """A venue rejected a request or behaved unexpectedly."""


class CredentialsNotVerified(VenueError):
    """Raised when live execution is attempted before credentials were proven.

    Exists because the WEEX authentication scheme in this codebase is inferred
    rather than confirmed against reachable documentation. Placing a real order
    with an unverified signing implementation risks sending a malformed request
    that is either rejected or, worse, interpreted differently than intended.
    `tradingbot verify-credentials` must pass first.
    """
