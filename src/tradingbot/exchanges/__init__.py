"""Exchange adapters.

Two separate jobs, deliberately kept apart:

* Binance public data -> deep, clean history for backtesting and feature
  computation. No account needed.
* WEEX -> contract specifications, funding rates, live quotes and (later)
  execution, because that is where the trades actually go.

On a 4h timeframe the price difference between venues is noise, but a signal is
validated against the WEEX mark price before it is ever acted on.
"""

from .binance import BinanceHistory
from .weex import WeexClient, WeexEndpointError

__all__ = ["WeexClient", "WeexEndpointError", "BinanceHistory"]
