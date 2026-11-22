# trading-bot

A crypto **signal-analysis** bot. It analyses the market and tells you what it
would do, with entry, stop, target and reasoning. It does not place orders.

Governing design rule: **the LLM does judgment, code does math.** Indicators,
levels, position sizing and risk checks are deterministic Python. The reasoning
layer only ever receives an already-computed snapshot.

## Status: Phase 0, partially built

Written so far:

| Path | What it does |
|---|---|
| `config/universe.yaml` | The 6 symbols and their tiers. Changing what is traded is a config edit, not a code edit. |
| `config/risk.yaml` | Deterministic risk rules with veto power over the reasoning layer. |
| `src/tradingbot/models.py` | `Candle`, `ContractSpec`, `SymbolConfig`, tiers. Decimal everywhere that touches sizing. |
| `src/tradingbot/config.py` | YAML loading, typed accessors, `live_trading_enabled()` kill switch. |
| `src/tradingbot/exchanges/weex.py` | Contract-spec loader (probes v1/v2/v3 paths, normalises field names, caches 24h) and klines. |
| `src/tradingbot/exchanges/binance.py` | Paged OHLCV history for backtesting. Drops the in-progress bar to avoid lookahead bias. |

Not yet written: `data/store.py` (candle persistence), `journal/` (signal log +
schema), `cli.py`, tests, and the Phase 1 screener. `git init` has not been run.

## Universe

| Symbol | Tier | Role |
|---|---|---|
| BTCUSDT | core | Regime anchor |
| SOLUSDT | core | High beta, real liquidity |
| NILUSDT | satellite | User pick |
| GOATUSDT | satellite | User pick, thin book |
| DOGEUSDT | satellite | High beta, deep book, long history |
| AXISUSDT | **paper** | ~1 month of history, ambiguous ticker. Signals journaled, never sized. Auto-promotes after 90 days of stored candles. |

Plus ETHUSDT and BTC dominance as context-only feeds.

## Two things to verify first

**1. The real WEEX minimums.** I could not reach `api-contract.weex.com` from
my environment, so the endpoint paths and field names in `weex.py` are
unverified — that is exactly why it probes several and keeps the raw payload.
Run this in **PowerShell** (not cmd.exe, which has no `$HOME` or `Write-Host`):

```powershell
curl.exe -s "https://api-contract.weex.com/capi/v3/market/contracts" -o "$HOME\Downloads\weex-contracts.json"; Write-Host "Saved $((Get-Item "$HOME\Downloads\weex-contracts.json").Length) bytes"
```

If that saves 0 bytes, swap `v3` for `v2`. The cmd.exe equivalent is:

```cmd
curl.exe -s "https://api-contract.weex.com/capi/v3/market/contracts" -o "%USERPROFILE%\Downloads\weex-contracts.json" && dir "%USERPROFILE%\Downloads\weex-contracts.json"
```

**2. Minimums are an exchange floor, not a per-coin one.** If WEEX enforces ~$5
minimum order value, it applies to GOATUSDT as much as BTCUSDT. Low-caps do not
lower your minimum — leverage lowers your *margin*. $5 notional at 10x costs
$0.50. `ContractSpec.min_cost()` computes this per pair from live data.

## Install

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Then copy `.env.example` to `.env`. `LIVE_TRADING_ENABLED` stays `false`; there
is no execution path in Phase 0 and it is gated from the first commit.

## Roadmap

0. **Data + journal** (in progress) — spec loader, candle store, signal log.
1. **Deterministic screener + backtester.** Produces the rules-only baseline.
   *This is the benchmark the LLM version has to beat.*
2. **LLM analyst** on screener candidates. Structured JSON out, never prose.
3. **Risk gate + Telegram delivery.**
4. **Multi-agent debate.** Measured against Phase 1. It may not win.
5. **Forward paper test, 3 months minimum.**
6. **Self-review loop** over the journal.

Phase 1 beating Phase 4 is a plausible outcome. Building in this order means
finding that out in week 3 rather than month 8.

## Backtest honesty

An LLM asked about SOL in March 2025 already knows what happened. That
contamination is the main reason paper backtests look brilliant and live
trading does not. Mitigations, strongest first: forward-test on unseen data;
restrict backtests to periods after the model cutoff; anonymise instrument and
date in the prompt. Do the first one regardless.

This is a personal analysis tool, not financial advice, and publishing its
signals to others raises licensing questions in most jurisdictions.
