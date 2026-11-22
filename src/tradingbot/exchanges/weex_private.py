"""WEEX authenticated API: quotes, funding, orders, positions.

**Read this before enabling live trading.**

The authentication scheme here is *inferred*, not confirmed. `api-contract.weex.com`
was unreachable from the environment this was written in, so the signing format
below follows the OKX/Bitget family convention that WEEX's contract API is
modelled on:

    signature = base64(hmac_sha256(secret, timestamp + METHOD + path + body))

sent as `ACCESS-KEY`, `ACCESS-SIGN`, `ACCESS-TIMESTAMP`, `ACCESS-PASSPHRASE`.
That is a well-founded guess - the presence of a passphrase alongside key and
secret in `.env.example` is itself strong evidence of this family - but it is
still a guess.

Everything in this module is therefore built so that being wrong is **safe and
obvious** rather than silent and expensive:

1. `verify_credentials()` performs a read-only authenticated call. Live
   execution refuses to start until it has passed, so a signing mismatch
   surfaces as a clear error on a harmless endpoint rather than on an order.
2. Several signing variants are implemented and can be selected by config, so
   adapting to the real scheme is a setting rather than a rewrite.
3. The raw response is retained on every call, so a field-name mismatch is
   visible in the journal instead of becoming a silently-missing stop.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

import httpx

from ..models import (
    FundingInfo,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Quote,
    Side,
)
from .base import CredentialsNotVerified, VenueError, VenuePosition
from .weex import WeexClient, WeexEndpointError, _first, _to_decimal, _unwrap

log = logging.getLogger(__name__)

SignStyle = Literal["okx", "bitget_query", "hex"]

# Candidate endpoint paths, probed the same way the public client probes.
TICKER_PATHS = ("/capi/v3/market/ticker", "/capi/v2/market/ticker")
DEPTH_PATHS = ("/capi/v3/market/depth", "/capi/v2/market/depth")
FUNDING_PATHS = (
    "/capi/v3/market/funding-rate",
    "/capi/v3/market/fundingRate",
    "/capi/v2/market/funding-rate",
)
ACCOUNT_PATHS = ("/capi/v3/account/accounts", "/capi/v2/account/accounts")
POSITION_PATHS = ("/capi/v3/position/allPosition", "/capi/v2/position/allPosition")
ORDER_PATH = "/capi/v3/order/placeOrder"
CANCEL_PATH = "/capi/v3/order/cancel_order"

_BID_KEYS = ("bestBid", "bid", "buyPrice", "bid1", "bidPx")
_ASK_KEYS = ("bestAsk", "ask", "sellPrice", "ask1", "askPx")
_MARK_KEYS = ("markPrice", "mark", "last", "lastPrice", "close", "indexPrice")
_FUNDING_KEYS = ("fundingRate", "funding_rate", "rate", "currentFundingRate")
_EQUITY_KEYS = ("equity", "accountEquity", "totalEquity", "usdtEquity", "available")


@dataclass
class WeexTradingClient(WeexClient):
    """WEEX client with authenticated endpoints.

    Inherits the public client's endpoint probing and field normalisation so
    the two halves cannot drift apart.
    """

    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    sign_style: SignStyle = "okx"
    # Flipped only by a successful verify_credentials(). Guards every
    # order-placing call.
    _verified: bool = field(default=False, init=False)

    # -- signing -----------------------------------------------------------

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)

    def _timestamp(self) -> str:
        """Milliseconds since epoch, as a string.

        Most venues in this family reject a timestamp more than 30 seconds from
        their own clock. A cron runner with a skewed clock would therefore fail
        every authenticated call, which is worth knowing when debugging.
        """
        return str(int(time.time() * 1000))

    def _sign(
        self, timestamp: str, method: str, path: str, body: str = "", query: str = ""
    ) -> str:
        if not self.api_secret:
            raise VenueError("Cannot sign a request without WEEX_API_SECRET")

        if self.sign_style == "bitget_query":
            message = f"{timestamp}{method.upper()}{path}{query}{body}"
        else:
            message = f"{timestamp}{method.upper()}{path}{body}"

        digest = hmac.new(
            self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).digest()
        if self.sign_style == "hex":
            return digest.hex()
        return base64.b64encode(digest).decode("utf-8")

    def _auth_headers(
        self, method: str, path: str, body: str = "", query: str = ""
    ) -> dict[str, str]:
        if not self.has_credentials:
            raise VenueError(
                "WEEX credentials are incomplete. Set WEEX_API_KEY, "
                "WEEX_API_SECRET and WEEX_API_PASSPHRASE."
            )
        timestamp = self._timestamp()
        return {
            "ACCESS-KEY": self.api_key or "",
            "ACCESS-SIGN": self._sign(timestamp, method, path, body, query),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.api_passphrase or "",
            "Content-Type": "application/json",
            "locale": "en-US",
        }

    def _signed_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """One authenticated call.

        The body is serialised once and the *same string* is both signed and
        sent. Re-serialising would risk a different key order or spacing, and
        the signature would not match the bytes on the wire - a genuinely
        maddening bug to diagnose.
        """
        body = json.dumps(payload, separators=(",", ":")) if payload else ""
        query = ""
        if params:
            query = "?" + "&".join(f"{key}={value}" for key, value in sorted(params.items()))

        headers = self._auth_headers(method, path, body, query)
        try:
            response = self._http().request(
                method.upper(),
                path + (query if method.upper() == "GET" else ""),
                content=body or None,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise VenueError(f"Transport error calling {path}: {exc}") from exc

        if response.status_code in (401, 403):
            raise VenueError(
                f"WEEX rejected authentication on {path} (HTTP {response.status_code}). "
                f"The signing scheme may differ from the assumed '{self.sign_style}' "
                f"style. Response: {response.text[:300]}"
            )
        if response.status_code != 200:
            raise VenueError(
                f"WEEX returned HTTP {response.status_code} on {path}: {response.text[:300]}"
            )
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise VenueError(f"Non-JSON response from {path}: {response.text[:200]}") from exc

    def verify_credentials(self) -> tuple[bool, str]:
        """Prove the signing scheme works, using a read-only endpoint.

        Must pass before any order can be placed. Tries each signing variant so
        that a mismatch produces an actionable answer ("use sign_style=X")
        rather than a bare failure.
        """
        if not self.has_credentials:
            return False, "WEEX credentials are not set"

        errors: list[str] = []
        for style in ("okx", "bitget_query", "hex"):
            self.sign_style = style  # type: ignore[assignment]
            for path in ACCOUNT_PATHS:
                try:
                    payload = self._signed_request("GET", path)
                except VenueError as exc:
                    errors.append(f"{style} {path}: {exc}")
                    continue
                if _unwrap(payload) or isinstance(payload, dict):
                    self._verified = True
                    return True, (
                        f"verified using sign_style={style} against {path}"
                    )
        return False, "no signing variant succeeded:\n  " + "\n  ".join(errors[:6])

    # -- public market data ------------------------------------------------

    def quote(self, symbol: str) -> Quote | None:
        """Top of book plus mark price.

        Returns None rather than raising: the risk gate treats a missing quote
        as a veto, so a venue hiccup becomes "no trade" instead of a crashed
        cron run.
        """
        try:
            _path, payload = self._try_paths(TICKER_PATHS, params={"symbol": symbol})
        except WeexEndpointError as exc:
            log.warning("No WEEX ticker endpoint responded for %s: %s", symbol, exc)
            return None

        entries = _unwrap(payload)
        entry = entries[0] if entries else (payload if isinstance(payload, dict) else {})
        if not isinstance(entry, dict):
            return None

        bid = _to_decimal(_first(entry, _BID_KEYS))
        ask = _to_decimal(_first(entry, _ASK_KEYS))
        mark = _to_decimal(_first(entry, _MARK_KEYS))

        if mark is None and bid is not None and ask is not None:
            mark = (bid + ask) / 2
        if bid is None or ask is None:
            # Without both sides there is no spread to check, and the gate
            # requires one. Falling back to the depth endpoint is better than
            # fabricating a spread of zero.
            return self._quote_from_depth(symbol, mark)
        if mark is None:
            mark = (bid + ask) / 2
        if bid <= 0 or ask <= 0 or ask < bid:
            log.warning("Nonsensical WEEX quote for %s: bid=%s ask=%s", symbol, bid, ask)
            return None

        return Quote(
            symbol=symbol, bid=bid, ask=ask, mark=mark, at=datetime.now(tz=UTC)
        )

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

    def funding(self, symbol: str) -> FundingInfo | None:
        """Current funding rate, normalised to a percentage per 8h.

        Venues report this inconsistently as a fraction (0.0001) or a percent
        (0.01). A value below 0.01 in absolute terms is almost certainly a
        fraction, since a genuine 0.01% funding rate is unremarkable while a
        1% one would be extraordinary.
        """
        try:
            _path, payload = self._try_paths(FUNDING_PATHS, params={"symbol": symbol})
        except WeexEndpointError:
            log.warning("No WEEX funding endpoint responded for %s", symbol)
            return None

        entries = _unwrap(payload)
        entry = entries[0] if entries else (payload if isinstance(payload, dict) else {})
        if not isinstance(entry, dict):
            return None

        raw_rate = _to_decimal(_first(entry, _FUNDING_KEYS))
        if raw_rate is None:
            return None
        rate_pct = raw_rate * Decimal("100") if abs(raw_rate) < Decimal("0.01") else raw_rate

        return FundingInfo(symbol=symbol, rate_8h_pct=rate_pct)

    # -- authenticated: account and positions ------------------------------

    def account_equity(self) -> Decimal | None:
        for path in ACCOUNT_PATHS:
            try:
                payload = self._signed_request("GET", path)
            except VenueError as exc:
                log.debug("Account path %s failed: %s", path, exc)
                continue
            entries = _unwrap(payload) or (
                [payload] if isinstance(payload, dict) else []
            )
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                equity = _to_decimal(_first(entry, _EQUITY_KEYS))
                if equity is not None:
                    return equity
        return None

    def open_positions(self) -> list[VenuePosition]:
        """Positions as WEEX reports them. The source of truth for reconciliation."""
        for path in POSITION_PATHS:
            try:
                payload = self._signed_request("GET", path)
            except VenueError as exc:
                log.debug("Position path %s failed: %s", path, exc)
                continue

            positions: list[VenuePosition] = []
            for entry in _unwrap(payload):
                quantity = _to_decimal(
                    _first(entry, ("total", "size", "positionQty", "holding", "available"))
                )
                if quantity is None or quantity == 0:
                    continue
                raw_side = str(
                    _first(entry, ("holdSide", "side", "posSide", "direction")) or ""
                ).lower()
                side = (
                    Side.SHORT
                    if any(token in raw_side for token in ("short", "sell", "2"))
                    else Side.LONG
                )
                entry_price = _to_decimal(
                    _first(entry, ("averageOpenPrice", "avgPrice", "openPrice", "costPrice"))
                )
                symbol = str(
                    _first(entry, ("symbol", "contract_code", "contractCode")) or ""
                )
                positions.append(
                    VenuePosition(
                        symbol=symbol.upper().replace("-", "").replace("_", ""),
                        side=side,
                        quantity=abs(quantity),
                        entry_price=entry_price or Decimal("0"),
                        unrealised_pnl=_to_decimal(
                            _first(entry, ("unrealizedPL", "unrealisedPnl", "upl"))
                        ),
                        leverage=_to_decimal(_first(entry, ("leverage", "lever"))),
                        raw=entry,
                    )
                )
            return positions
        return []

    # -- authenticated: orders ---------------------------------------------

    @property
    def is_paper(self) -> bool:
        return False

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Place one order.

        Refuses unless `verify_credentials()` has succeeded in this process.
        Given the signing scheme is inferred, sending an unverified order is
        the one failure this module exists to prevent.
        """
        if not self._verified:
            raise CredentialsNotVerified(
                "Refusing to place a live order before verify_credentials() has "
                "passed. Run `tradingbot verify-credentials` first: the WEEX "
                "signing scheme in this codebase is inferred, not confirmed."
            )

        payload: dict[str, Any] = {
            "symbol": request.symbol,
            "size": str(request.quantity),
            "client_oid": request.client_order_id,
            "type": _weex_order_type(request),
            "order_type": "0",  # 0 = normal; not post-only, not FOK
            "match_price": "1" if request.order_type is OrderType.MARKET else "0",
        }
        if request.price is not None:
            payload["price"] = str(request.price)
        if request.trigger_price is not None:
            payload["trigger_price"] = str(request.trigger_price)
        if request.leverage is not None:
            payload["leverage"] = str(request.leverage)
        if request.reduce_only:
            payload["reduce_only"] = "1"

        try:
            response = self._signed_request("POST", ORDER_PATH, payload=payload)
        except VenueError as exc:
            # A transport failure is NOT a safe "did not happen". The order may
            # have reached the venue. Reporting FAILED (not REJECTED) tells the
            # reconciler it must check the venue before retrying.
            return OrderResult(
                client_order_id=request.client_order_id,
                status=OrderStatus.FAILED,
                error=str(exc),
            )

        container = response if isinstance(response, dict) else {}
        data = container.get("data") if isinstance(container.get("data"), dict) else container
        venue_id = _first(data or {}, ("orderId", "order_id", "ordId")) if data else None

        code = str(container.get("code", "")).strip()
        if code and code not in ("0", "00000", "200", "success"):
            return OrderResult(
                client_order_id=request.client_order_id,
                status=OrderStatus.REJECTED,
                error=f"WEEX code {code}: {container.get('msg') or container.get('message')}",
                raw=container,
            )

        return OrderResult(
            client_order_id=request.client_order_id,
            status=OrderStatus.SUBMITTED,
            venue_order_id=str(venue_id) if venue_id else None,
            raw=container,
        )

    def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        try:
            response = self._signed_request(
                "POST",
                CANCEL_PATH,
                payload={"symbol": symbol, "client_oid": client_order_id},
            )
        except VenueError as exc:
            log.warning("Cancel failed for %s: %s", client_order_id, exc)
            return False
        code = str(response.get("code", "")) if isinstance(response, dict) else ""
        return code in ("", "0", "00000", "200", "success")


def _weex_order_type(request: OrderRequest) -> str:
    """Map a side plus reduce_only to the venue's position-action code.

    This family of APIs encodes direction and intent together:
      1 = open long, 2 = open short, 3 = close long, 4 = close short.

    Getting this wrong is the most dangerous possible mapping error: sending
    "open short" where "close long" was meant doubles exposure instead of
    flattening it. Hence it is one small function with one job.
    """
    if request.reduce_only:
        return "3" if request.side is Side.LONG else "4"
    return "1" if request.side is Side.LONG else "2"
