# trading-bot

A crypto **signal-analysis** bot for a small account. It analyses the market,
decides what it would do, explains why, sends it to you on Telegram, and then
scores its own prediction so you find out whether it was right.

**It does not place orders.** There are no exchange credentials anywhere in the
project and no code that could submit one. That is a structural guarantee
rather than a setting, and CI asserts it on every push so a future refactor
that reintroduces execution fails loudly.

Governing design rule: **the LLM does judgment, code does math.** Indicators,
structure, levels, patterns, position sizing and risk checks are deterministic
Python. The reasoning layer only ever receives an already-computed snapshot.

---

## How it is deployed, and why that shaped the code

This runs entirely on **free infrastructure**, which is a design constraint
rather than a footnote.

| Piece | Service | The free-tier reality |
|---|---|---|
| Database | **Neon** Postgres | 1 GB/project, 100 CU-hours/mo, compute suspends after 5 idle minutes |
| Scheduler | **GitHub Actions cron** | Public repo: unlimited minutes. Private: 2,000/mo as a **hard stop** |
| API + bot | **Render** free web service | 750 instance-hours/mo *per workspace*, sleeps after 15 min idle |
| Keep-alive | **cron-job.org** | 60 runs/hour maximum (a 1-minute floor), 30s timeout, 4-40s jitter |
| Interface | **Telegram bot** | Free, push-based, and it is the dashboard |

There is no free always-on worker anywhere as of late 2026: Fly removed its
allowance for new orgs and Render's background workers are paid-only. This bot
does not need one. Its primary decision timeframe is **4h**, so it needs six
scheduled runs a day, which is what cron is for.

### Keeping the service awake without waking the database

A free Render service sleeps after 15 minutes idle and takes ~60s to wake,
which would make every Telegram command feel broken. So an external pinger
(cron-job.org) hits `/ping` every ~10 minutes. Two details matter.

**`/ping` touches no database.** That is the whole reason it exists as a
separate endpoint rather than pinging `/health/db`. Neon's free plan allows
100 CU-hours/month and its compute suspends after 5 idle minutes, so a pinger
aimed at a database-backed endpoint would hold the compute on permanently and
exhaust the allowance. Keeping Render awake and letting Neon sleep are both
requirements, and they conflict unless the ping endpoint is deliberately inert.

**Always-on consumes nearly the whole Render allowance.** 750 instance-hours is
per *workspace* and a 31-day month is 744 hours, leaving about 6 hours of
margin. The interval does not change that: Render bills wall-clock time while
the service runs, not per request, so a 10-minute ping costs exactly what a
1-minute ping would. If the hours do run out, every free web service in the
workspace is suspended until the next month.

That is survivable by design, because **the scheduler does not depend on
Render.** The GitHub Actions workflow talks to Neon and Telegram directly, so
signals keep arriving even if the API is suspended. Only the interactive
commands go dark.

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

### How the bot scores itself

A journal of predictions nobody checks is just a diary. In `paper` mode every
approved signal opens a simulated position, and the monitor job later walks the
real candles printed since entry to decide whether the stop or the target was
reached. No venue is involved at any point.

Two honesty rules, because self-scoring is where this usually goes wrong:

**If one bar contains both the stop and the target, the stop is assumed to have
filled.** OHLC data cannot tell you which was touched first, and on a 4h bar
that ambiguity is common rather than rare. Assuming the favourable outcome is
exactly how a paper record manufactures an edge that does not survive a real
market. These are recorded as `stop_same_bar`, so you can count how many
results rest on the assumption.

**Entry slippage and round-trip fees are deducted.** A target that should pay
2.0R reports about **+1.94R**, and a stop reports about **-1.03R**. Those are
the honest numbers; a gross-only R-multiple overstates every small-account
result.

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
| `execution/` | Paper position tracking and candle-based resolution |
| `jobs/` | Ingest, analysis cycle, paper resolution |
| `notify/` | Telegram delivery **and the interactive bot** |
| `api/` | Read-only JSON API, the Telegram webhook, and `/ping` |
| `cli.py` | Every job as a command, plus `doctor` |
| `tests/` | 177 tests, no database, network or credentials required |

**Not built:** the LLM reasoning layer (Phase 2 scaffolding exists but is not
wired in) and the backtester.

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

### 4. Telegram (required, not optional)

It is the only delivery channel *and* the only interface, so without it the bot
runs silently.

1. Create a bot with `@BotFather` and copy the token.
2. Message your new bot, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and read your chat id.
3. Generate a webhook secret:
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`
4. Put all three in `.env`, then check delivery works:

```powershell
tradingbot telegram-test
```

### 5. Deploy the scheduler

Push to GitHub, then add under **Settings - Secrets and variables - Actions**:

| Kind | Name | Value |
|---|---|---|
| Secret | `DATABASE_URL` | pooled Neon string |
| Secret | `DATABASE_DIRECT_URL` | direct Neon string |
| Secret | `TELEGRAM_BOT_TOKEN` | from BotFather |
| Secret | `TELEGRAM_CHAT_ID` | your chat id |
| Variable | `TRADING_MODE` | `paper` |

Mode is a repository **variable**, not a secret, deliberately: which mode the
bot is in should be visible in the UI rather than hidden. There are no exchange
credentials to add.

Run the workflow manually once before trusting the schedule:
**Actions - trade - Run workflow - doctor**.

### 6. Deploy the API and the bot

Point Render at the repo; `render.yaml` describes the service. In the Render
dashboard set `DATABASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`TELEGRAM_WEBHOOK_SECRET` and `PUBLIC_BASE_URL` (your `https://....onrender.com`
URL).

Then register the webhook, which only works against the public HTTPS URL:

```powershell
tradingbot telegram-setup
```

Check it took with `GET /telegram/status`, which reports whether the webhook is
registered and shows Telegram's last delivery error without exposing the token.

### 7. Keep it awake

At [console.cron-job.org](https://console.cron-job.org), create one job:

| Setting | Value |
|---|---|
| URL | `https://your-app.onrender.com/ping` |
| Schedule | every **10 minutes** |

Ten minutes, not one. The free plan's floor is 1 minute and its jitter is
4-40 seconds, but Render's idle timeout is 15 minutes, so 10 gives comfortable
margin at a fraction of the requests. It costs no more Render hours either,
since those are billed by wall-clock time while running.

**Point it at `/ping`, not `/health/db` or `/status`.** Anything that queries
the database would hold Neon's compute awake around the clock and burn the
100 CU-hour monthly allowance.

## The Telegram bot is the interface

There is no web frontend, and that is a deliberate choice rather than a gap.
A signal matters the moment the 4h bar closes; a dashboard requires you to go
and look, and on a free host it needs a minute to wake up before it can show
you anything. Telegram pushes.

It also removes a whole layer of work: no separate deploy, no CORS, no
authentication to build, no session handling. Telegram already knows who you
are.

Approved signals arrive automatically. These commands are for inspecting what
the bot has been doing:

```
/status                 equity, open paper trades, circuit breaker
/signals [n]            the last n verdicts
/journal [n]            verdicts with the reason each was approved or rejected
/performance [days]     results in R, with expectancy
/positions              currently tracked paper trades
/analyse BTCUSDT        analyse one symbol right now
/universe               what is traded, and the risk rules
/runs                   recent scheduled runs, to check the scheduler is firing
/help                   list commands
```

`/runs` is worth knowing about: it warns when the last run is more than ~8
hours old, which is how you notice a silently stopped scheduler. A bot that has
found no setups looks identical to a bot that is dead.

**Webhook, not polling.** Long-polling needs a process that runs forever and
there is no free always-on worker, so the bot is a plain HTTP endpoint on the
API service that already exists.

**Two independent access barriers**, because a webhook URL is effectively
public and the bot can read your journal and trigger analysis:

1. Telegram echoes a shared secret in `X-Telegram-Bot-Api-Secret-Token`, which
   is verified before the body is parsed. **If `TELEGRAM_WEBHOOK_SECRET` is
   unset, the webhook rejects everything** rather than accepting everything.
2. The sender's chat id must be in the allow-list. An unconfigured bot answers
   nobody.

Both failures return HTTP 200. Telegram disables a webhook that keeps erroring,
so a non-2xx response would turn one bad update into an outage, and a 200 also
avoids confirming to a prober that the endpoint is real.

## One thing to verify first

The exchange adapters have never been run against a live endpoint, because the
environment this was built in had no network access.

**The WEEX public endpoints.** `exchanges/weex.py` probes several documented
paths and normalises field names, keeping the raw payload so a mismatch is
visible rather than silent. Only public routes are used now, so there is no
signing scheme to get wrong and no credentials to hold.

Check it saves something:

```powershell
curl.exe -s "https://api-contract.weex.com/capi/v3/market/contracts" -o "$HOME\Downloads\weex-contracts.json"; Write-Host "Saved $((Get-Item "$HOME\Downloads\weex-contracts.json").Length) bytes"
```

If that saves 0 bytes, try `v2`. Then run `tradingbot refresh-specs`.

Why WEEX at all when candles come from Binance: the risk gate's friction checks
need the spread and funding of the venue a trade would actually be taken on.
Binance's spread tells you nothing about WEEX's book, and on a thin satellite
pair the spread is what destroys the edge.

Note that minimums are an **exchange floor, not a per-coin one**. If WEEX
enforces a ~$5 minimum order value it applies to GOATUSDT as much as BTCUSDT.
Low caps do not lower your minimum; leverage lowers your *margin*. $5 notional
at 10x costs $0.50 of margin. `ContractSpec.min_cost()` computes this per pair
from live data.

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
exchanges/     Binance (history) + WEEX public (specs, quotes, funding)
      |                                 no credentials, no order routes
jobs/ingest    resumable candle sync, per-timeframe retention
      |
db/            Neon Postgres: candles, signals, positions, breakers
      |
analysis/      indicators -> structure -> levels -> patterns -> regime
                            -> confluence -> FeatureSnapshot
      |
strategy/      entry, stop, targets from real structure
      |
risk/          sizing (Decimal) + the veto gate  <-- holds final authority
      |
execution/     paper position opened; later resolved against real candles
      |
notify/        Telegram: push delivery + the interactive bot
```

Two independent runtimes, deliberately:

* **GitHub Actions** runs ingest, analysis and resolution. It needs only Neon
  and Telegram, so it keeps working if the API is down or suspended.
* **Render** serves the Telegram webhook and the JSON API. Losing it costs the
  interactive commands and nothing else.

Each layer can veto the next. The risk gate's authority is absolute: no score
and no future LLM output can override it.

### Design decisions worth knowing

**Idempotency everywhere.** The scheduler can fire twice. Analysis refuses to
re-process a bar it has completed; `client_order_id` is a deterministic hash of
(symbol, bar, side, purpose), so a replayed run is rejected both locally and by
the venue.

**No execution capability at all.** Not a disabled feature, an absent one:
no credentials in the settings, no authenticated endpoints in the exchanges
package, no order-placing code. CI asserts all three on every push, so the
guarantee cannot quietly erode through a later refactor.

**Rejections are first-class journal rows.** A bot that only records its
approvals cannot tell you whether its filters are too tight, and that is
usually the most valuable thing in the log.

**The gate fails closed.** If the spread cannot be fetched, the signal is
rejected in paper mode. Assuming a benign default would mean the one time
WEEX's ticker is down is the one time a signal is scored as though friction
were free.

**Paper resolution is pessimistic.** Same-bar stop-and-target resolves to the
stop, and fills carry slippage and fees. The point of paper mode is to learn
whether the analysis works, and an optimistic simulator cannot answer that.

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

4. **Position-close ordering was non-deterministic.** Several positions can
   close in the same microsecond, and `consecutive_losses()` stops counting at
   the first win, so the losing streak - and the circuit-breaker decision built
   on it - depended on an arbitrary tie order.

The first two were paper/live divergences, the category to be most suspicious
of: they make paper results stop predicting real behaviour. Both were fixed
before live execution was removed, and both still matter, because paper results
are now the only evidence the bot produces about itself.

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
8. **Self-review loop** over the journal.

Execution is deliberately not on this list, and neither is a web frontend:
Telegram is the interface.

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
tradingbot monitor              resolve paper positions, check breakers
tradingbot status               equity, positions, recent runs
tradingbot journal [-n 50]      recent signals with their reasons
tradingbot clear-breaker        manually clear a circuit breaker
tradingbot refresh-specs        refresh cached WEEX contract minimums
tradingbot telegram-setup       register the webhook and command menu
tradingbot telegram-test        send a test message
tradingbot close-paper SYM      close a tracked paper position in the journal
```

---

This is a personal analysis tool, not financial advice. It reports what it
would do; every actual trade is your own decision, placed by hand. Publishing
its signals to others raises licensing questions in most jurisdictions.
