# trading-bot

A crypto **signal-analysis and execution** bot for a small account. It analyses
the market, decides what it would do, explains why, and — once explicitly armed
— places the trade with its stop resting at the exchange.

Governing design rule: **the LLM does judgment, code does math.** Indicators,
structure, levels, patterns, position sizing and risk checks are deterministic
Python. The reasoning layer only ever receives an already-computed snapshot.

---

## How it is deployed, and why that shaped the code

This runs on **free infrastructure**, which is a design constraint rather than
a footnote. The single most important consequence:

> **The bot is asleep most of the time.** It wakes on a scheduler that can be
> delayed by 5–15 minutes. It cannot watch price, cannot react to a move, and
> must never be the only thing standing between an open position and a loss.

Everything below follows from that.

| Piece | Service | The free-tier reality |
|---|---|---|
| Database | **Neon** Postgres | 1 GB/project, 100 CU-hours/mo, compute suspends after 5 idle minutes |
| Scheduler | **GitHub Actions cron** | Public repo: unlimited minutes. Private: 2,000/mo as a **hard stop** |
| API | **Render** free web service | 750 instance-hours/mo, sleeps after 15 min idle, ~60s cold start |
| Frontend | **Vercel** | Free for a personal project (not built yet) |
| Delivery | **Telegram** | Free, push-based, works while everything else sleeps |

There is no free always-on worker anywhere as of late 2026 — Fly removed its
allowance for new orgs and Render's background workers are paid-only. This bot
does not need one: its primary decision timeframe is **4h**, so it needs six
scheduled runs a day, which is what cron is for.

### The minute budget is real

`.github/workflows/trade.yml` documents this inline, but the headline:

```
analyse   6 runs/day  × ~2 min    ≈ 360 min/month
monitor  24 runs/day  × ~0.4 min  ≈ 290 min/month   (exits in seconds when flat)
                                  ───────────────
                                   ~650 min/month
```

That fits inside a private repo's 2,000 minutes with room for CI. A 15-minute
analysis cadence would cost **~5,760 min/month** and die around day 10 of every
month, silently. If you want a faster cadence, make the repo public — Actions
minutes are then unlimited.

### The consequence for live execution

A cron that can be delayed cannot manage a stop-loss. So:

**Every entry places its protective stop as a real reduce-only order on WEEX,
in the same operation as the entry.** The exchange enforces it 24/7. If the
stop cannot be placed, the engine **closes the position immediately** rather
than leaving it unprotected. An open position with no resting stop is treated
as an incident: the monitor job flattens it and Telegram alerts you.

---

## Status

Phases 0–3 are built and tested. Phase 4+ are not started.

| Layer | What is there |
|---|---|
| `settings.py` | Env-driven config; two-switch live-trading gate |
| `db/` | Neon-tuned engine, schema with idempotent natural keys, repositories |
| `analysis/` | Indicators, market structure, S/R + volume profile, candlestick and chart patterns, divergence, regime/beta, confluence scoring, `FeatureSnapshot` |
| `strategy/` | Entry/stop/target construction from structure |
| `risk/` | Decimal sizing, the veto gate, circuit breakers |
| `execution/` | Paper broker and live engine sharing one code path |
| `jobs/` | Ingest, analysis cycle, monitor/reconcile |
| `api/` | Read-mostly FastAPI over the journal |
| `cli.py` | Every job as a command, plus `doctor` |
| `tests/` | 133 tests, no database or network required |

**Not built:** the LLM reasoning layer (Phase 2 scaffolding exists but is not
wired in), the backtester, and the frontend.

**Not verified:** the exchange adapters have never been run against a live
endpoint, because the environment this was built in had no network access. See
*Two things to verify first* below.

---

## The analysis engine

Deterministic throughout. Every number is computed by hand-written code — no
TA-Lib, both because it needs C extensions free hosts cannot build and because
an indicator you cannot unit-test against a hand-computed value is one you
cannot trust.

**Indicators** — EMA/SMA, Wilder smoothing, ATR, RSI, MACD, Bollinger,
Keltner, Donchian, ADX/DI, SuperTrend, Stochastic, StochRSI, Williams %R, CCI,
OBV, MFI, VWAP, relative volume.

**Structure** — fractal swing detection, higher-high/lower-low sequences,
break of structure and change of character, EMA stack, range position.

**Levels** — swing clustering with ATR-scaled tolerance (so the same code works
for BTC at five figures and a token at four decimals), touch counts, flipped
levels, volume profile with POC and value area, round numbers.

**Patterns** — hammer, shooting star, engulfing, momentum bar, doji, inside
bar; double top/bottom, head-and-shoulders and its inverse, ascending/
descending/symmetrical triangles, rising/falling wedges, channels; RSI and
MACD divergence (regular and hidden).

**Regime** — BTC trend and volatility, ETH/BTC rotation, realised beta to BTC
from log returns. Satellite longs are blocked outright when the backdrop is
hostile.

**Confluence** — each check returns a signed contribution in `[-1, 1]` with a
weight; the score is the weighted mean, so adding a check cannot inflate the
scale. Separately, a set of **blockers** can veto a trade no matter how high
the score, because some setups are wrong regardless of how good the rest of the
chart looks.

### Three lookahead guards worth knowing about

These are the difference between a real backtest and a fiction, and each is
pinned by a test:

1. **Swings confirm late.** A swing high needs N bars of hindsight, so at the
   moment it printed it was not yet a swing. Every `Swing` carries
   `confirmed_at`, and `confirmed_swings()` is the only accessor the strategy
   layer may use.
2. **Donchian excludes the current bar.** Without the shift, "price broke the
   20-bar high" is true by construction and every breakout backtest looks
   brilliant.
3. **Warm-up periods stay NaN.** A backfilled indicator value is lookahead
   bias wearing a disguise.

---

## Setup

### 1. Local

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
Copy-Item .env.example .env
```

### 2. Neon

Create a free project at neon.tech, then copy **both** connection strings into
`.env`. They are not interchangeable:

* `DATABASE_URL` — the **pooled** string (host contains `-pooler`). Used by the
  app and every cron job; the pooler absorbs the reconnect when Neon's compute
  wakes from suspend.
* `DATABASE_DIRECT_URL` — the unpooled string, for schema changes only.
  PgBouncer in transaction mode cannot run DDL reliably.

```powershell
tradingbot init-db
tradingbot doctor      # run this before anything else
```

### 3. First data load and a dry run

```powershell
tradingbot ingest      # first run backfills; later runs are incremental
tradingbot analyse     # signal_only mode: nothing is ordered
tradingbot journal     # see what it decided and why
```

### 4. Telegram (optional, recommended)

Create a bot via `@BotFather`, then get your chat id by messaging it and
opening `https://api.telegram.org/bot<TOKEN>/getUpdates`. It is the only push
channel, and the API sleeps on the free tier.

### 5. Deploy the scheduler

Push to GitHub, then add under **Settings → Secrets and variables → Actions**:

| Kind | Name | Value |
|---|---|---|
| Secret | `DATABASE_URL` | pooled Neon string |
| Secret | `DATABASE_DIRECT_URL` | direct Neon string |
| Secret | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | optional |
| Secret | `WEEX_API_KEY` / `_SECRET` / `_PASSPHRASE` | only for paper/live |
| Variable | `TRADING_MODE` | `signal_only` to start |
| Variable | `LIVE_TRADING_ENABLED` | `false` |

Mode is a repository **variable**, not a secret, deliberately — which mode the
bot is in should be visible in the UI rather than hidden.

Then run the workflow manually once (`Actions → trade → Run workflow → doctor`)
before trusting the schedule.

### 6. Deploy the API

Point Render at the repo; `render.yaml` describes the service. Set
`DATABASE_URL` in the dashboard. The service hardcodes `signal_only`, so the
API can never place an order.

---

## Arming live execution

Four independent things must all be true. This is deliberate friction.

1. `TRADING_MODE=live`
2. `LIVE_TRADING_ENABLED=true` — either alone does nothing
3. All three WEEX credentials present — settings refuse to load otherwise
4. `tradingbot verify-credentials` passes — **the engine refuses to place any
   order until it has**

Step 4 exists because the WEEX authentication scheme in this codebase is
*inferred*, not confirmed (see below). Verification runs against a read-only
endpoint, so a signing mismatch surfaces harmlessly instead of on an order.

**Do not skip paper mode.** The paper broker shares the live execution code
path and models fees and adverse slippage (wider for thin satellite books), so
its results are indicative rather than flattering. The README's own advice
still stands: forward-test for months before risking money.

---

## Two things to verify first

Neither could be checked while building this, because the build environment
had no network access.

**1. The real WEEX endpoints and signing scheme.** `exchanges/weex.py` probes
several documented paths and normalises field names, keeping the raw payload so
a mismatch is visible rather than silent. `exchanges/weex_private.py` assumes
the OKX/Bitget-family signature:

```
base64(hmac_sha256(secret, timestamp + METHOD + path + body))
```

sent as `ACCESS-KEY` / `ACCESS-SIGN` / `ACCESS-TIMESTAMP` / `ACCESS-PASSPHRASE`.
The presence of a passphrase alongside key and secret is strong evidence of
this family, but it is still a guess. Three variants are implemented and
selectable; `verify-credentials` tries each and tells you which works.

Check the public endpoint first:

```powershell
curl.exe -s "https://api-contract.weex.com/capi/v3/market/contracts" -o "$HOME\Downloads\weex-contracts.json"; Write-Host "Saved $((Get-Item "$HOME\Downloads\weex-contracts.json").Length) bytes"
```

If that saves 0 bytes, try `v2`.

**2. Minimums are an exchange floor, not a per-coin one.** If WEEX enforces a
~$5 minimum order value, it applies to GOATUSDT as much as BTCUSDT. Low-caps do
not lower your minimum — leverage lowers your *margin*. $5 notional at 10x
costs $0.50 of margin. `ContractSpec.min_cost()` computes this per pair from
live data.

---

## Known issue: two symbols can never trade

`risk.yaml` contains a contradiction, left in place because it is a risk
decision rather than a bug to be silently "fixed":

| Symbol | `beta_to_btc` | `max_portfolio_btc_beta` | Result |
|---|---|---|---|
| BTCUSDT | 1.0 | 2.0 | fine |
| SOLUSDT | 1.5 | 2.0 | fine |
| DOGEUSDT | 1.8 | 2.0 | fine |
| NILUSDT | 2.0 | 2.0 | tradeable only alone |
| **GOATUSDT** | **2.5** | 2.0 | **never tradeable** |
| **AXISUSDT** | **2.5** | 2.0 | **never tradeable** |

Each of those two exceeds the entire portfolio beta budget by itself, so the
gate vetoes them even on an empty book. Related: **no two tradeable symbols can
be held in the same direction**, which makes `max_open_positions: 2`
unreachable for same-direction trades.

`tradingbot doctor` reports this, and a test pins it so a config change is
noticed. Your options:

* Raise `max_portfolio_btc_beta` to ~3.0 — two positions become possible.
* Lower the satellite betas — but 2.5 for a thin-book low-cap is honest.
* Accept it, and treat GOATUSDT and AXISUSDT as analysis-only.

### A second config implication

`min_risk_reward: 2.0` is checked **after** friction, as `risk.yaml` intends.
With the configured fee, spread and funding assumptions, a setup needs roughly
**2.2+ gross** R:R to clear a 2.0 net floor. On a small account friction is not
a rounding error. Expect few signals; that is the filter working.

---

## Architecture

```
exchanges/     Binance (history) + WEEX (specs, quotes, funding, orders)
      |
jobs/ingest    resumable candle sync, per-timeframe retention
      |
db/            Neon Postgres: candles, signals, orders, positions, breakers
      |
analysis/      indicators -> structure -> levels -> patterns -> regime
                            -> confluence -> FeatureSnapshot
      |
strategy/      entry, stop, targets from real structure
      |
risk/          sizing (Decimal) + the veto gate  <-- holds final authority
      |
execution/     entry + reduce-only stop, atomically in intent
      |
notify/        Telegram
```

Each layer can veto the next. The risk gate's authority is absolute: no score
and no future LLM output can override it.

### Design decisions worth knowing

**Idempotency everywhere.** The scheduler can fire twice. Analysis refuses to
re-process a bar it has completed; `client_order_id` is a deterministic hash of
(symbol, bar, side, purpose), so a replayed run is rejected both locally and by
the venue.

**Order rows are written before the network call.** If the process dies
mid-request, the next run finds a `PENDING` row and reconciles against the
venue. Writing after the call could leave a live position the bot does not know
about — the worst failure mode available to it.

**Rejections are first-class journal rows.** A bot that only records its
approvals cannot tell you whether its filters are too tight, and that is
usually the most valuable thing in the log.

**The gate fails closed.** If the spread cannot be fetched, the trade is
rejected. Assuming a benign default means the one time WEEX's ticker is down is
the one time an order goes into a blown-out book.

**R-multiples, not percentages.** The only honest way to compare a BTC trade
with a GOATUSDT trade.

---

## Three bugs the tests found

Recorded because each is a class of error worth watching for, and each is now
pinned by a regression test.

1. **The paper broker rejected market `reduce_only` closes.** Such an order
   carries no price, so the broker had no reference to fill against — meaning
   an emergency close that works live *failed in paper*, in exactly the
   scenario the execution engine exists to handle.
2. **The quote never reached the venue object.** Paper fills were priced from
   the position's own entry, so every closed paper trade showed a P&L of
   roughly zero no matter what the market did. Paper mode was measuring
   nothing.
3. **A parabolic move was classified as a range.** A near-vertical rally
   produces *no* fractal pivots — no bar has lower highs on both sides — so the
   swing sequence is empty precisely when the trend is strongest. A rally with
   ADX 99 was labelled `RANGE`, which zeroed the trend contribution, disarmed
   the strong-trend blockers, and re-enabled the mean-reversion check that is
   meant to switch itself off in a trend. The bot was primed to fade a
   parabolic move.

The first two were paper/live divergences, which is the category to be most
suspicious of: they make paper results stop predicting live behaviour.

---

## Roadmap

0. **Data + journal** — done
1. **Deterministic screener** — done. *This is the baseline the LLM must beat.*
2. **LLM analyst** on screener candidates. Structured JSON out, never prose.
   Scaffolding exists (`FeatureSnapshot.to_prompt_dict`), not wired in.
3. **Risk gate + Telegram delivery** — done
4. **Backtester** over the stored candles, sharing the live analysis path.
5. **Multi-agent debate.** Measured against Phase 1. It may not win.
6. **Forward paper test, 3 months minimum.**
7. **Self-review loop** over the journal.
8. **Frontend** (Next.js on Vercel, consuming the API).

Phase 1 beating Phase 5 is a plausible outcome. Building in this order means
finding that out in week 3 rather than month 8.

---

## Backtest honesty

An LLM asked about SOL in March 2025 already knows what happened. That
contamination is the main reason paper backtests look brilliant and live
trading does not. Mitigations, strongest first: forward-test on unseen data;
restrict backtests to periods after the model cutoff; anonymise instrument and
date in the prompt. `FeatureSnapshot.to_prompt_dict(anonymise=True)` implements
the third and is tested to strip the symbol. Do the first one regardless.

## Commands

```
tradingbot doctor               check the deployment is wired up correctly
tradingbot init-db              create tables
tradingbot ingest               sync candles from Binance
tradingbot analyse [--force]    run one analysis cycle
tradingbot monitor              reconcile positions, check breakers
tradingbot status               account, positions, recent runs
tradingbot journal [-n 50]      recent signals with their reasons
tradingbot verify-credentials   prove WEEX signing works (required for live)
tradingbot clear-breaker        manually clear a circuit breaker
tradingbot close-position SYM   close a position at market
```

---

This is a personal analysis tool, not financial advice. Publishing its signals
to others raises licensing questions in most jurisdictions.
