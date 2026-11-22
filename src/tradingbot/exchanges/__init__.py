"""Exchange adapters.

Two separate jobs, deliberately kept apart:

* Binance public data -> deep, clean history for backtesting and feature
  computation. No account needed.
* WEEX public data -> contract specifications, funding rates and live quotes,
  because that is the book a trade would actually be taken on and its spread is
  what erodes the edge.

No authenticated endpoints and no credentials: the bot analyses and reports, it
does not place orders. On a 4h timeframe the cross-venue price difference is
noise, but a signal is still sanity-checked against the WEEX mark price so a
stale or divergent setup is caught.
"""

from .base import MarketDataSource, VenueError
from .binance import BinanceHistory
from .weex import WeexClient, WeexEndpointError
from .weex_market import WeexMarketData, fetch_market_data

__all__ = [
    "BinanceHistory",
    "MarketDataSource",
    "VenueError",
    "WeexClient",
    "WeexEndpointError",
    "WeexMarketData",
    "fetch_market_data",
]
