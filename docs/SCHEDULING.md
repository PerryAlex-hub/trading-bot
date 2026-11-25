# Scheduling: why GitHub cron is not a clock

## The measurement

On 2026-11-25, with one cron per job:

| Job | Expected | Delivered | Rate |
|---|---|---|---|
| analyse (6/day) | 5 | 2 | **40%** |
| monitor (hourly) | ~22 | 3 | **14%** |

Every run that fired **succeeded**. Both workflows were `active`. GitHub simply
dropped the triggers — it queues scheduled workflows and discards them under
load, and it documents the delay but not the drops.

At one point the last analyse run had been **eight hours earlier** on a
schedule that promises every four.

## What is already done about it, in-repo

**1. Missed bars are recovered, not lost.** `analyse` begins with
`run_catch_up`, which walks back up to 6 bars, finds any with no successful
run, and analyses each with the candle series truncated at that bar. A dropped
trigger therefore costs *timeliness*, not data. This is verified in production:

```
Backfilling missed bar 2026-11-25 12:00:00+00:00
Catch-up: backfilled 1 bar(s) [11-25 12:00]
```

**2. Three attempts per bar instead of one.** A duplicate run is free:
`run_cycle` checks `RunRepo.already_ran` for the bar and returns *before* the
Telegram notifier is constructed, so a second or third attempt at a finished
bar costs about a minute of Actions time and sends nothing. Actions minutes are
unlimited on a public repository.

A 4h bar with `open_time` 20:00 closes at 00:00, so runs at 00:23, 00:53 and
01:13 all see 20:00 as the most recently closed bar. The window is 70 minutes
wide and every attempt in it is equivalent. If each trigger lands 40% of the
time, at least one lands about **78%** of the time.

**3. Off the contended minutes.** Nothing runs in the first ten minutes of an
hour, which is the busiest slot on the platform because so many crons are
written on the hour.

**4. You are told when it slips.** The bot warns if the last run is older than
~5h on a 4h schedule, so a silent stall is visible rather than looking like a
market with no setups.

## The real fix: an external clock

Three unreliable triggers are better odds, not a guarantee. To make the cadence
deterministic, drive `workflow_dispatch` from something whose job is being a
clock, and let GitHub be only the executor. You already run a cron-job.org
pinger to keep the Render service awake, and the same free account can do this.

### 1. Create a fine-grained token

GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
tokens → **Generate new token**.

- **Repository access**: Only select repositories → `PerryAlex-hub/trading-bot`
- **Permissions**: Repository permissions → **Actions: Read and write**
- Nothing else. This token can start workflows in one repository and do
  nothing else, which is the whole point of using a fine-grained one.
- Set an expiry you will actually renew. 90 days is reasonable.

Keep it out of this repository. It goes into cron-job.org only.

### 2. Add the cron-job.org job

Create a job per schedule you want. For the analysis:

- **URL**: `https://api.github.com/repos/PerryAlex-hub/trading-bot/actions/workflows/trade.yml/dispatches`
- **Method**: `POST`
- **Schedule**: `0,4,8,12,16,20` hours at minute `25` (UTC)
- **Headers**:
  - `Accept: application/vnd.github+json`
  - `Authorization: Bearer <your token>`
  - `X-GitHub-Api-Version: 2022-11-28`
  - `Content-Type: application/json`
- **Body**:

```json
{"ref": "master", "inputs": {"job": "analyse"}}
```

For the monitor, the same with `monitor.yml` in the URL and
`{"ref": "master"}` as the body — that workflow takes no inputs.

A successful dispatch returns **HTTP 204 with an empty body**. cron-job.org
will show that as a success. A 404 almost always means the token lacks Actions
write on this repository rather than a wrong URL.

### 3. Then thin out the in-repo crons

Once the external trigger is landing reliably, the three redundant GitHub
schedules become noise. Drop back to one per bar as a fallback and let the
external clock set the cadence. Do not remove them entirely: the fallback is
what keeps signals arriving if the cron-job.org account lapses or the token
expires.

## Verifying it

```powershell
# What actually ran, and when
gh run list --workflow trade.yml --limit 10

# Did the catch-up recover anything?
gh run view <run-id> --log | Select-String "Catch-up|Backfilling"
```

`gh` lives at `C:\Program Files\GitHub CLI\gh` and may not be on PATH.
