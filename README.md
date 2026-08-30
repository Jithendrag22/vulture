# Vulture — 4H crypto paper-trade scanner

Donchian breakout + 200EMA + volatility gate on 45 USDT pairs, 4H bars.
Signals go to Telegram. **Places no orders.** It is an alerter and a journal.

## Where it runs

GitHub Actions, on a schedule. Nothing needs to stay switched on.

`fapi.binance.com` returns **HTTP 451** to a GitHub-hosted runner (measured:
Azure westus, San Jose). `data-api.binance.vision` answers 200 with identical
symbols, naming and JSON — but serves **spot**, not perpetuals.

`RESULTS_SPOT_PARITY.md` measures whether that matters. On a matched period,
across all 8 Donchian variants:

| | futures | spot |
|---|---|---|
| live config (lookback 55, atr 2.0, rr 3.0) | +0.0849R | +0.0762R |
| out-of-sample | +0.1043R | +0.0907R |
| expectancy correlation | | **0.9943** |
| rank ordering | | unchanged |

The futures series also **fails** the harness's random-entry control
(+0.0432R, t=3.22) while spot **passes** it (+0.0128R, t=0.97) over the same
dates — so the ~10% lower number may be the honest one rather than a cost.

## Scheduling

Bars close at 00 04 08 12 16 20 UTC. GitHub's cron is best-effort and skips
runs under load, so this fires **three times per bar** and lets the scanner
decide:

```
scanner.py --if-new-bar
```

It compares the exchange's latest closed bar against `paper_trades/last_bar.txt`
and exits in seconds if there is nothing new. A late run still catches the bar;
a duplicate does nothing. All three would have to be dropped to miss one.

## State

`paper_trades/journal.jsonl` is committed back by the workflow after each scan.
The repo is the state store — durable, and readable as a history.

## Secrets

`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Running locally

Defaults to Binance futures, so a local run is unchanged:

```
python scanner.py                    # one scan, futures
VULTURE_MARKET=spot python scanner.py --if-new-bar
python scanner.py --loop             # long-running, wakes at each bar close
```

## Known limitations

- Signals come from **spot** prices; the paper trade is sized as a perpetual.
- The liquidation recorder is not here. It is a persistent WebSocket and Actions
  jobs cap at six hours.
- Paper only. No broker is connected.
