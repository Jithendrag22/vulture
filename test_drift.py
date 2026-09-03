"""The entry price in an alert is the CLOSED BAR's close, but the scan runs
after it - 5 minutes later at best, ~65 if GitHub drops both earlier attempts.
These pin the staleness reporting, in both directions and for both sides.

A sign error here would be actively dangerous: it would tell you a chased
entry was a favourable one."""
import os
import sys
from unittest.mock import patch

os.environ.setdefault("VULTURE_MARKET", "spot")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanner  # noqa: E402

BASE = dict(symbol="UNIUSDT", signal_time="2026-08-30 00:00:00+00:00",
            target=5.81, atr_pct=3.07, atr_rank=0.70, risk_pct_move=6.14,
            notional_usdt=185, qty=37.7, rr=3.0)
ENTRY = 4.906
RISK = 0.301


def drift(side, now, stop):
    a = scanner.Alert(side=side, entry=ENTRY, stop=stop, **BASE)
    with patch.object(scanner, "http_json", return_value={"price": str(now)}):
        return scanner.live_drift(a)


def test_long_chased_is_flagged():
    out = drift("long", ENTRY + 0.5 * RISK, 4.605)
    assert "+0.50R" in out and "chased" in out


def test_long_favourable_is_flagged():
    out = drift("long", ENTRY - 0.5 * RISK, 4.605)
    assert "-0.50R" in out and "better than quoted" in out


def test_short_inverts_correctly():
    """For a SHORT, a price FALL is adverse - you are selling lower."""
    adverse = drift("short", ENTRY - 0.5 * RISK, 5.207)
    assert "+0.50R" in adverse and "chased" in adverse
    better = drift("short", ENTRY + 0.5 * RISK, 5.207)
    assert "-0.50R" in better and "better than quoted" in better


def test_small_drift_carries_no_flag():
    out = drift("long", ENTRY + 0.05 * RISK, 4.605)
    assert "chased" not in out and "better than quoted" not in out


def test_price_lookup_failure_never_breaks_the_alert():
    a = scanner.Alert(side="long", entry=ENTRY, stop=4.605, **BASE)
    with patch.object(scanner, "http_json", side_effect=OSError("network down")):
        assert scanner.live_drift(a) == ""


def test_age_is_measured_from_bar_close_not_open():
    """signal_time is the bar's OPEN time; a 4H bar closes 4h later.

    Ageing from the open reported every alert as 4h staler than it really was,
    which for a catch-up scan is the difference between "act now" and "stale".
    """
    from datetime import datetime, timedelta, timezone

    opened = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)      # closes 12:00
    now = opened + timedelta(hours=4, minutes=7)                  # 7 min post-close

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    a = scanner.Alert(side="long", entry=ENTRY, stop=4.605,
                      **{**BASE, "signal_time": str(opened)})
    with patch.object(scanner, "http_json", return_value={"price": str(ENTRY)}), \
         patch.object(scanner, "datetime", _Clock):
        line = scanner.live_drift(a)

    assert "7 min ago" in line, line
    assert "247 min ago" not in line, "aged from the bar open, not its close"


def test_mark_to_market_does_not_persist_a_synthetic_journal():
    """A what-if check must never overwrite the real record.

    mark_to_market rewrites journal.jsonl from the list it is handed. Called
    with a test fixture and no guard, it replaces live trading history with
    that fixture -- which is exactly how the Mac journal was destroyed.
    """
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    # A bar window the fixture's signal_time sits inside, so the position
    # really does close and the write path really is reached.
    bars = scanner.fetch_klines("BTCUSDT", "4h",
                                limit=scanner.MAX_HOLD_BARS + scanner.MAX_CATCHUP_BARS + 5)
    sig = bars["dt"].iloc[1]
    px = float(bars["close"].iloc[1])

    with TemporaryDirectory() as tmp:
        real = Path(tmp) / "journal.jsonl"
        keep = {"symbol": "REALUSDT", "status": "open"}
        real.write_text(json.dumps(keep) + "\n")

        fake = [dict(symbol="BTCUSDT", side="long", status="open",
                     signal_time=str(sig), entry=px,
                     stop=px * 0.99, target=px * 1.01)]   # both reachable fast
        with patch.object(scanner, "journal_path", return_value=real):
            notes = scanner.mark_to_market(fake, persist=False)

        assert fake[0]["status"] == "closed", "fixture must reach the write path"
        assert [json.loads(l) for l in real.read_text().splitlines()] == [keep]
