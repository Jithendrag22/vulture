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
