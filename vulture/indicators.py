#!/usr/bin/env python3
"""Vectorised indicators. All strictly causal - no value uses future bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l = df["high"], df["low"]
    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    a = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / a
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / a

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std(ddof=0)
    return ma - k * sd, ma, ma + k * sd


def bb_width(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    lo, mid, hi = bollinger(series, n, k)
    return (hi - lo) / mid


def zscore(series: pd.Series, n: int) -> pd.Series:
    mu = series.rolling(n).mean()
    sd = series.rolling(n).std(ddof=0)
    return (series - mu) / sd.replace(0, np.nan)


def rolling_percentile(series: pd.Series, n: int) -> pd.Series:
    """Percentile rank of the CURRENT value within the trailing n-window."""
    return series.rolling(n).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True
    )


def donchian(df: pd.DataFrame, n: int):
    """Prior-N-bar extremes, EXCLUDING the current bar (shift(1))."""
    return df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def realised_vol(series: pd.Series, n: int = 24) -> pd.Series:
    return np.log(series / series.shift(1)).rolling(n).std(ddof=0)


def wick_ratios(df: pd.DataFrame):
    """Lower and upper wick as a fraction of the bar's total range."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_lo = df[["open", "close"]].min(axis=1)
    body_hi = df[["open", "close"]].max(axis=1)
    return (body_lo - df["low"]) / rng, (df["high"] - body_hi) / rng


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard indicator set used across strategies."""
    out = df.copy()
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["close"]
    out["rsi14"] = rsi(out["close"], 14)
    out["rsi2"] = rsi(out["close"], 2)
    out["adx14"] = adx(out, 14)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["bbw"] = bb_width(out["close"], 20, 2.0)
    out["bbw_pct"] = rolling_percentile(out["bbw"], 100)
    out["ret1"] = out["close"].pct_change()
    out["rvol"] = out["volume"] / out["volume"].rolling(20).mean()
    out["range_atr"] = (out["high"] - out["low"]) / out["atr14"]
    out["lower_wick"], out["upper_wick"] = wick_ratios(out)
    out["vol_z"] = zscore(out["volume"], 100)
    out["ret_z"] = zscore(out["ret1"], 100)
    return out
