#!/usr/bin/env python3
"""
Vulture live scanner + paper-trading alerter.

Runs at 4H bar close, evaluates the surviving strategy across the universe,
and emits fully-specified trade instructions (entry / SL / TP / size) to
Telegram. Every signal is logged to a paper-trade journal and marked to market
on subsequent runs.

THIS PLACES NO ORDERS. It is an alerter and a journal, nothing more.

The point of this phase is not to make money. It is to generate the clean,
never-seen-by-the-optimiser data needed to validate the volatility gate, whose
backtested improvement is contaminated by selection.

Setup:
    1. Talk to @BotFather on Telegram -> /newbot -> copy the token
    2. Message your new bot once, then visit
       https://api.telegram.org/bot<TOKEN>/getUpdates to find your chat id
    3. export TELEGRAM_BOT_TOKEN=...   export TELEGRAM_CHAT_ID=...

Usage:
    python3 scanner.py --dry-run       # print alerts, do not send
    python3 scanner.py                 # send to Telegram
    python3 scanner.py --loop          # run forever, waking at each 4H close
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

from vulture import indicators

ROOT = Path(__file__).resolve().parent
JOURNAL = ROOT / "paper_trades"
JOURNAL.mkdir(exist_ok=True)


def load_env(path: Path = ROOT / ".env") -> None:
    """
    Read KEY=VALUE lines from .env into the environment without overwriting
    anything already set. Keeps the token out of shell history and out of the
    process list.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


load_env()

# --- market / data source ------------------------------------------------
#
# MEASURED: fapi.binance.com returns HTTP 451 to a GitHub-hosted runner
# (Azure westus, San Jose). api.binance.com does too. data-api.binance.vision
# answers 200 with identical symbols, naming and JSON - but it serves SPOT.
#
# RESULTS_SPOT_PARITY.md measures whether that matters: on a matched period the
# strategy's expectancy correlates 0.9943 between the two series, the rank
# ordering of every variant is unchanged, and the live config goes +0.0849R ->
# +0.0762R. The futures series also FAILS the random-entry control (+0.043R,
# t=3.22) while spot passes it, so the lower number may be the honest one.
#
# VULTURE_MARKET=spot selects the mirror. Default stays futures so nothing
# changes for a local run.
MARKET = os.environ.get("VULTURE_MARKET", "futures").lower()
if MARKET == "spot":
    KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
else:
    KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

# 45 symbols. The 30 added beyond the original 15 were never used in
# strategy development and hold up at t=+6.03 independently - a
# cross-sectional out-of-sample check. Survives 2x fees + 3x slippage.
UNIVERSE = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "APTUSDT", "ARBUSDT",
    "ATOMUSDT", "AVAXUSDT", "AXSUSDT", "BNBUSDT", "BTCUSDT",
    "COMPUSDT", "CRVUSDT", "DOGEUSDT", "DOTUSDT", "EOSUSDT",
    "ETCUSDT", "ETHUSDT", "FILUSDT", "FTMUSDT", "GALAUSDT",
    "GRTUSDT", "HBARUSDT", "ICPUSDT", "INJUSDT", "LDOUSDT",
    "LINKUSDT", "LTCUSDT", "MANAUSDT", "MATICUSDT", "NEARUSDT",
    "OPUSDT", "RUNEUSDT", "SANDUSDT", "SEIUSDT", "SNXUSDT",
    "SOLUSDT", "SUIUSDT", "SUSHIUSDT", "THETAUSDT", "TIAUSDT",
    "TRXUSDT", "UNIUSDT", "VETUSDT", "XLMUSDT", "XRPUSDT",
]

# ---- Strategy parameters. Frozen from the backtest. Do not tune live. ----
#
# Two strategies run side by side. They are only worth running together
# because their monthly returns correlate at just 0.32 - when one is in a bad
# patch the other frequently is not. Hybrid backtest: Rs 42,577/yr vs
# Rs 34,958/yr for Donchian alone, with LOWER drawdown (9.0% vs 9.5%).
INTERVAL = "4h"
MAX_HOLD_BARS = 30       # 5 days. Chosen on WORST-YEAR floor (Rs 31,272) not mean.
RR = 3.0                 # target 3x risk. 8x has a higher mean but a LOWER median.
ATR_STOP = 2.0

# Strategy 1: Donchian breakout + 200EMA + volatility regime gate
LOOKBACK = 55
ATR_RANK_MIN = 0.5

# Strategy 2: 52-week-high momentum (Journal of Banking & Finance 2025)
# Price near its yearly high, making a new 20-bar high. Long only, by
# construction - the anchoring effect it exploits has no short analogue.
H52_WINDOW = 2190        # bars of 4H = 365 days
H52_NEAR = 0.95          # within 5% of the 52-week high
H52_BREAKOUT = 20        # and making a new 20-bar high
H52_COOLDOWN_BARS = 6    # don't re-fire on the same push

# ---- Account parameters ----
# Account size and risk come from the ENVIRONMENT, never from the source.
# The repository is public; the strategy is textbook and worth publishing, the
# operator's account size is not. Set VULTURE_CAPITAL / VULTURE_RISK_PCT (as
# repository secrets in CI, or in .env locally). The defaults below are a
# placeholder so a clone runs and produces sane-looking sizing - they are not
# anybody's real numbers.
CAPITAL = float(os.environ.get("VULTURE_CAPITAL", 50_000.0))     # rupees
RISK_PCT = float(os.environ.get("VULTURE_RISK_PCT", 0.01))       # 1% default
MAX_CONCURRENT = 5        # tested 1-10; 5 beat 3 in 6/7 years (+Rs 13,114/yr, DD 8.8->9.2%)
USDINR = 88.0              # refreshed manually; only affects display sizing

# Slippage buffer applied to the quoted entry so a late manual fill is still
# inside the tested envelope.
ENTRY_BUFFER = 0.0005      # 0.05%

STRATEGY_LABEL = {
    "donchian": "Donchian-55 + 200EMA + vol gate",
    "high52": "52-week-high momentum",
}


@dataclass
class Alert:
    symbol: str
    side: str
    signal_time: str
    entry: float
    stop: float
    target: float
    atr_pct: float
    atr_rank: float
    risk_pct_move: float
    notional_usdt: float
    qty: float
    strategy: str = "donchian"
    rr: float = RR
    status: str = "open"
    exit_time: str | None = None
    exit_price: float | None = None
    outcome: str | None = None
    pnl_r: float | None = None


# ---------------------------------------------------------------------------

def http_json(url: str, timeout=20):
    req = Request(url, headers={"User-Agent": "vulture-scanner/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_klines(symbol: str, interval: str, limit: int = 400) -> pd.DataFrame:
    raw = http_json(f"{KLINES_URL}?symbol={symbol}&interval={interval}&limit={limit}")
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # Drop the still-forming final bar - signals are only valid on CLOSED bars.
    return df.iloc[:-1].reset_index(drop=True)


def fetch_daily_context(symbol: str, days: int = 365) -> dict:
    """
    One daily-bar request serving two purposes:
      - the trailing 52-week high (Strategy 2)
      - the daily EMA50 trend, used as a multi-timeframe filter on BOTH
        strategies. Requiring the daily trend to agree with the 4H signal
        removes only 7% of signals but lifted expectancy +0.118R -> +0.152R
        and cut drawdown 11.9% -> 9.3%, improving in 7 of 7 years.
    Returns {} when there is not enough listed history to judge either.
    """
    try:
        d = fetch_klines(symbol, "1d", limit=min(days + 5, 1000))
    except Exception:
        return {}
    if len(d) < 200:
        return {}
    ema50 = d["close"].ewm(span=50, adjust=False).mean()
    return {
        "high52": float(d["high"].iloc[-days:].max()),
        "daily_uptrend": bool(d["close"].iloc[-1] > ema50.iloc[-1]),
    }


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - skipping send")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urlencode({"chat_id": chat, "text": text,
                      "parse_mode": "HTML", "disable_web_page_preview": "true"}).encode()
    try:
        with urlopen(Request(url, data=data), timeout=20) as r:
            resp = json.loads(r.read().decode())
        if not resp.get("ok"):
            # Telegram answers 200 with ok=false for bad chat ids, blocked bots
            # and rate limits. Silently returning False here is how a dead
            # channel looks identical to a healthy one in the log.
            print(f"[telegram] API rejected the send: {resp}")
            return False
        return True
    except (URLError, HTTPError) as e:
        print(f"[telegram] send failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------

def _build_alert(d, i, symbol, side, strategy, atr, rank, extra=None) -> Alert:
    c = float(d["close"].iloc[i])
    stop_dist = ATR_STOP * atr
    if side == "long":
        entry = c * (1 + ENTRY_BUFFER)
        stop, target = entry - stop_dist, entry + RR * stop_dist
    else:
        entry = c * (1 - ENTRY_BUFFER)
        stop, target = entry + stop_dist, entry - RR * stop_dist

    risk_rupees = CAPITAL * RISK_PCT
    risk_move = stop_dist / entry
    notional_usdt = (risk_rupees / USDINR) / risk_move

    return Alert(
        symbol=symbol, side=side,
        signal_time=str(d["dt"].iloc[i]),
        entry=round(entry, 6), stop=round(stop, 6), target=round(target, 6),
        atr_pct=round(float(d["atr_pct"].iloc[i]) * 100, 3),
        atr_rank=round(rank, 3),
        risk_pct_move=round(risk_move * 100, 3),
        notional_usdt=round(notional_usdt, 2),
        qty=round(notional_usdt / entry, 6),
        strategy=strategy,
    )


def evaluate(symbol: str) -> list[Alert]:
    """
    Evaluate BOTH strategies on the most recently CLOSED bar.

    Returns a list because the two can fire on the same symbol at the same
    time. When they do we keep only one - the same position twice is just
    double size, not diversification.
    """
    df = fetch_klines(symbol, INTERVAL, limit=400)
    if len(df) < 300:
        return []

    d = indicators.add_all(df[["dt", "open", "high", "low", "close", "volume"]])
    d["atr_pct_rank"] = indicators.rolling_percentile(d["atr_pct"], 200)

    i = len(d) - 1
    c = float(d["close"].iloc[i])
    a = float(d["atr14"].iloc[i])
    e200 = float(d["ema200"].iloc[i])
    rank = float(d["atr_pct_rank"].iloc[i])
    if not all(np.isfinite(x) for x in (c, a, e200, rank)):
        return []

    # Daily context: 52-week high AND the daily trend used as the MTF filter.
    # One request, two uses. If it is unavailable we do not trade the symbol
    # this bar rather than trading it unfiltered.
    ctx = fetch_daily_context(symbol)
    if not ctx:
        return []
    daily_up = ctx["daily_uptrend"]

    out: list[Alert] = []

    # --- Strategy 1: Donchian breakout, volatility-gated, MTF-confirmed ---
    if rank >= ATR_RANK_MIN:
        prior_hi = float(d["high"].iloc[i - LOOKBACK:i].max())
        prior_lo = float(d["low"].iloc[i - LOOKBACK:i].min())
        side = None
        if c > prior_hi and c > e200:
            side = "long"
        elif c < prior_lo and c < e200:
            side = "short"
        # MTF: the daily trend must agree with the direction of the trade
        if side and ((side == "long") == daily_up):
            out.append(_build_alert(d, i, symbol, side, "donchian", a, rank))

    # --- Strategy 2: 52-week-high momentum (long only) ---
    #
    # The yearly high comes from DAILY bars. 365 days of 4H would be 2,190
    # candles and Binance caps a klines request at 1,500, so asking on the 4H
    # endpoint silently returns a shorter window - turning "52-week high" into
    # "8-month high" with no error raised.
    hi52 = ctx["high52"]
    if daily_up and np.isfinite(hi52) and hi52 > 0:
        hi20 = float(d["high"].iloc[i - H52_BREAKOUT:i].max())
        if c / hi52 >= H52_NEAR and c > hi20:
            out.append(_build_alert(d, i, symbol, "long", "high52", a, rank))

    # De-duplicate: if both fired the same direction on this symbol, keep one.
    seen = set()
    unique = []
    for al in out:
        key = (al.symbol, al.side)
        if key in seen:
            continue
        seen.add(key)
        unique.append(al)
    return unique


def journal_path() -> Path:
    return JOURNAL / "journal.jsonl"


def load_journal() -> list[dict]:
    p = journal_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append_journal(a: Alert):
    with journal_path().open("a") as f:
        f.write(json.dumps(asdict(a), separators=(",", ":")) + "\n")


def open_positions(journal: list[dict]) -> list[dict]:
    return [j for j in journal if j.get("status") == "open"]


def mark_to_market(journal: list[dict]) -> list[str]:
    """Check open paper positions against current price; close on SL/TP."""
    notes = []
    changed = False
    for j in journal:
        if j.get("status") != "open":
            continue
        try:
            df = fetch_klines(j["symbol"], INTERVAL, limit=MAX_HOLD_BARS + 5)
        except Exception:
            continue
        sig_t = pd.Timestamp(j["signal_time"])
        after = df[df["dt"] > sig_t]
        if after.empty:
            continue

        for _, bar in after.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if j["side"] == "long":
                hit_stop, hit_tgt = lo <= j["stop"], hi >= j["target"]
            else:
                hit_stop, hit_tgt = hi >= j["stop"], lo <= j["target"]
            if hit_stop:  # pessimistic: stop first if both
                j.update(status="closed", outcome="stop", pnl_r=-1.0,
                         exit_time=str(bar["dt"]), exit_price=j["stop"])
                notes.append(f"❌ {j['symbol']} {j['side']} stopped out (-1R)")
                changed = True
                break
            if hit_tgt:
                j.update(status="closed", outcome="target", pnl_r=float(RR),
                         exit_time=str(bar["dt"]), exit_price=j["target"])
                notes.append(f"✅ {j['symbol']} {j['side']} hit target (+{RR}R)")
                changed = True
                break
        else:
            if len(after) >= MAX_HOLD_BARS:
                last = after.iloc[MAX_HOLD_BARS - 1]
                px = float(last["close"])
                risk = abs(j["entry"] - j["stop"])
                r = ((px - j["entry"]) if j["side"] == "long" else (j["entry"] - px)) / risk
                j.update(status="closed", outcome="time", pnl_r=round(r, 3),
                         exit_time=str(last["dt"]), exit_price=px)
                notes.append(f"⏱ {j['symbol']} {j['side']} time stop ({r:+.2f}R)")
                changed = True

    if changed:
        with journal_path().open("w") as f:
            for j in journal:
                f.write(json.dumps(j, separators=(",", ":")) + "\n")
    return notes


def safe_leverage(stop_pct_move: float, min_ratio: float = 3.0,
                  maint_margin: float = 0.005, cap: int = 5) -> int:
    """
    Highest leverage at which the liquidation price still sits at least
    `min_ratio` times further away than the stop.

    Liquidation on isolated margin is roughly (100/lev - maint%) adverse.
    Requiring liq_distance >= min_ratio * stop_distance gives:

        100/lev - 0.5 >= min_ratio * stop_pct
        lev <= 100 / (min_ratio * stop_pct + 0.5)

    Getting this wrong is how a correctly-sized trade still blows up: at 20x
    liquidation lands at ~4.5%, INSIDE a 6% stop, so the stop never fires.
    """
    denom = min_ratio * stop_pct_move + maint_margin * 100
    if denom <= 0:
        return 1
    return max(1, min(cap, int(100.0 / denom)))



def _ticker_url() -> str:
    base = KLINES_URL.rsplit("/", 1)[0]
    return f"{base}/ticker/price"


def live_drift(a: "Alert") -> str:
    """How stale the quoted entry is, in time and in R.

    The alert quotes the CLOSED BAR's close as the entry, and the backtest
    assumes entry there with zero delay. But the scan runs after the close -
    5 minutes later at best, and up to ~65 minutes if GitHub drops the first
    two scheduled attempts. By the time this is read the market has moved, and
    a percentage is hard to judge by eye. The useful unit is R: how much of the
    trade's own risk has already been given away before entering.
    """
    try:
        now = float(http_json(f"{_ticker_url()}?symbol={a.symbol}", timeout=10)["price"])
    except Exception:
        return ""     # never let a price lookup break the alert

    bar = datetime.fromisoformat(a.signal_time)
    age = (datetime.now(timezone.utc) - bar).total_seconds() / 60.0

    risk = abs(a.entry - a.stop)
    # Positive = ADVERSE: the move since the bar close is against the entry,
    # so the same stop now sits closer and the same target further away.
    adverse_r = ((now - a.entry) if a.side == "long" else (a.entry - now)) / risk
    pct = (now - a.entry) / a.entry * 100.0

    if adverse_r >= 0.25:
        flag = "  ⚠️ chased"
    elif adverse_r <= -0.25:
        flag = "  ✅ better than quoted"
    else:
        flag = ""

    return (f"Now     <code>{now:g}</code>   ({pct:+.2f}% vs entry · "
            f"{adverse_r:+.2f}R given up · bar closed {age:.0f} min ago){flag}\n")


def format_alert(a: Alert) -> str:
    arrow = "🟢 LONG" if a.side == "long" else "🔴 SHORT"
    risk_rupees = CAPITAL * RISK_PCT
    lev = safe_leverage(a.risk_pct_move)
    notional_inr = a.notional_usdt * USDINR
    margin_inr = notional_inr / lev
    liq_dist = (100.0 / lev) - 0.5

    return (
        f"<b>{arrow}  {a.symbol}</b>\n"
        f"<i>{STRATEGY_LABEL.get(a.strategy, a.strategy)} · 4H</i>\n\n"
        f"Entry   <code>{a.entry:g}</code>\n"
        f"{live_drift(a)}"
        f"Stop    <code>{a.stop:g}</code>   ({a.risk_pct_move:.2f}% away)\n"
        f"Target  <code>{a.target:g}</code>   (1:{a.rr:g})\n\n"
        f"<b>Size</b>   <code>{a.qty:g}</code> {a.symbol.replace('USDT','')}\n"
        f"       ≈ {a.notional_usdt:,.0f} USDT  /  ₹{notional_inr:,.0f} notional\n"
        f"<b>Lev</b>    <code>{lev}x isolated</code>  →  margin ₹{margin_inr:,.0f}\n"
        f"       liquidation ~{liq_dist:.0f}% away "
        f"({liq_dist/a.risk_pct_move:.1f}x your stop)\n\n"
        f"Risk    ₹{risk_rupees:,.0f} if stopped  ·  "
        f"reward ₹{risk_rupees*a.rr:,.0f} at target\n"
        f"ATR     {a.atr_pct:.2f}%  ·  vol rank {a.atr_rank:.2f}\n\n"
        f"<i>Paper trade. Bar close {a.signal_time[:16]}</i>"
    )


def summary_line(journal: list[dict]) -> str:
    closed = [j for j in journal if j.get("status") == "closed" and j.get("pnl_r") is not None]
    if not closed:
        return "No closed paper trades yet."
    pnl = np.array([j["pnl_r"] for j in closed], dtype=float)
    wins = (pnl > 0).sum()
    return (f"Paper record: {len(closed)} closed · {wins}W/{len(closed)-wins}L · "
            f"{wins/len(closed):.0%} WR · total {pnl.sum():+.2f}R · "
            f"expectancy {pnl.mean():+.3f}R")


def next_4h_close() -> datetime:
    now = datetime.now(timezone.utc)
    h = (now.hour // 4 + 1) * 4
    nxt = now.replace(minute=0, second=30, microsecond=0)
    return (nxt.replace(hour=0) + timedelta(days=1) if h >= 24
            else nxt.replace(hour=h))


def scan_once(dry_run: bool):
    journal = load_journal()

    notes = mark_to_market(journal)
    journal = load_journal()

    open_now = open_positions(journal)
    n_open = len(open_now)
    slots = max(0, MAX_CONCURRENT - n_open)

    # A symbol that already has an open position must not be entered again.
    # Slot counting alone did not prevent this: a re-trigger on a later bar,
    # or any duplicate scan of the same bar, opened a SECOND position in the
    # same name - doubling the risk on one symbol while the journal still
    # reported it as one of five independent slots. Found live on 2026-08-30,
    # which produced two UNIUSDT longs off the same 00:00 bar.
    held = {p["symbol"] for p in open_now}

    # 45 symbols x 2 endpoints is 90 requests; sequentially that is ~50s and a
    # single hung request stalls the whole scan. Fan out, and let one symbol's
    # failure be that symbol's problem only.
    alerts = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(evaluate, sym): sym for sym in UNIVERSE}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                alerts.extend(fut.result())
            except Exception as e:
                print(f"  {sym}: error {type(e).__name__}: {e}")

    # When slots are scarce, prefer the higher-expectancy strategy first
    # (52w-high momentum backtested +0.242R vs Donchian +0.119R), then the
    # strongest volatility regime.
    before = len(alerts)
    alerts = [a for a in alerts if a.symbol not in held]
    if before != len(alerts):
        notes.append(f"{before - len(alerts)} signal(s) suppressed - position "
                     f"already open in that symbol.")

    alerts.sort(key=lambda x: (0 if x.strategy == "high52" else 1, -x.atr_rank))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"<b>Vulture scan</b> · {ts}",
             f"Open paper positions: {n_open}/{MAX_CONCURRENT}"]
    if notes:
        lines.append("\n" + "\n".join(notes))

    taken = []
    if not alerts:
        lines.append("\nNo signals this bar.")
    else:
        for a in alerts[:slots]:
            taken.append(a)
        if slots == 0:
            lines.append(f"\n{len(alerts)} signal(s) fired but all slots are full — skipped.")
        elif len(alerts) > slots:
            lines.append(f"\n{len(alerts)} signals, {slots} slot(s) free — taking the "
                         f"highest volatility rank.")

    lines.append("\n" + summary_line(journal))
    header = "\n".join(lines)

    print(header.replace("<b>", "").replace("</b>", "")
          .replace("<i>", "").replace("</i>", ""))
    if not dry_run:
        send_telegram(header)

    for a in taken:
        msg = format_alert(a)
        print("\n" + msg.replace("<b>", "").replace("</b>", "")
              .replace("<i>", "").replace("</i>", "")
              .replace("<code>", "").replace("</code>", ""))
        if not dry_run:
            send_telegram(msg)
        append_journal(a)

    return taken



LAST_BAR = ROOT / "paper_trades" / "last_bar.txt"


def latest_closed_bar() -> datetime:
    """Open time of the most recent CLOSED 4H bar, from the exchange itself."""
    return fetch_klines("BTCUSDT", "4h", limit=3)["dt"].iloc[-1].to_pydatetime()


def new_bar_since_last_scan() -> tuple[bool, datetime]:
    """Has a 4H bar closed that we have not scanned?

    GitHub Actions cron is best-effort: runs arrive late and are sometimes
    skipped entirely under load. Scheduling the scan for an exact bar-close
    time therefore drops bars silently. Instead the workflow runs OFTEN and
    this decides whether there is anything to do, by comparing the exchange's
    latest closed bar against the last one we recorded. A late run still
    catches the bar; a duplicate run does nothing.
    """
    bar = latest_closed_bar()
    try:
        seen = datetime.fromisoformat(LAST_BAR.read_text().strip())
    except (OSError, ValueError):
        return True, bar
    return bar > seen, bar


def record_scanned_bar(bar: datetime) -> None:
    LAST_BAR.parent.mkdir(parents=True, exist_ok=True)
    LAST_BAR.write_text(bar.isoformat())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--if-new-bar", action="store_true",
                    help="scan only if a 4H bar has closed since the last scan")
    args = ap.parse_args()

    if args.if_new_bar:
        fresh, bar = new_bar_since_last_scan()
        if not fresh:
            print(f"No new 4H bar (latest closed {bar:%Y-%m-%d %H:%M} UTC). Nothing to do.")
            return
        print(f"New 4H bar closed at {bar:%Y-%m-%d %H:%M} UTC - scanning.", flush=True)
        scan_once(args.dry_run)
        record_scanned_bar(bar)
        return

    if not args.loop:
        scan_once(args.dry_run)
        return

    print("Loop mode: waking at each 4H bar close. Ctrl-C to stop.")
    while True:
        try:
            scan_once(args.dry_run)
        except Exception as e:
            print(f"scan error: {type(e).__name__}: {e}")
        nxt = next_4h_close()
        wait = (nxt - datetime.now(timezone.utc)).total_seconds()
        print(f"\nNext scan at {nxt:%Y-%m-%d %H:%M} UTC ({wait/3600:.1f}h)\n", flush=True)
        # Sleep against the wall clock in short chunks. time.sleep() does not
        # advance while macOS is suspended, so a single long sleep silently
        # drifts past the bar close every time the lid closes -- the scanner
        # stays alive and simply never fires. Chunking re-reads the real clock
        # after every wake, so a missed bar is caught within 30s of waking.
        while True:
            remaining = (nxt - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                if remaining < -300:
                    print(f"[wake] overslept the {nxt:%H:%M} UTC bar by "
                          f"{-remaining/60:.0f} min - scanning now.", flush=True)
                break
            time.sleep(min(30.0, remaining))


if __name__ == "__main__":
    main()
