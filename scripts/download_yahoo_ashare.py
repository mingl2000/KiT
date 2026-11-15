"""Rebuild the demo dataset layout from Yahoo Finance.

Writes, for every ticker in instruments.json:

    <out>/ashare/{1m,5m,15m,30m,1h,2h,1d}/<code>.parquet

with the same schema as the reference data in data/ashare/: columns
datetime (bar CLOSE time, naive Asia/Shanghai, datetime64[ns]), open, high, low,
close, volume (float64), a plain RangeIndex, bars on the A-share session grid
(09:30-11:30, 13:00-15:00) and back-adjusted prices. instruments.json,
stats.json and an extended trade_dates.csv are placed next to it, so <out> can
be passed straight to `--data-root`.

Yahoo specifics handled here:
  * codes: 600031.SH -> 600031.SS, 000001.SZ unchanged;
  * Yahoo stamps bars with their OPEN time; they are re-stamped to close times;
  * Yahoo's 60m bars (09:30, 10:30, 12:30, 13:30, 14:30) do not match the
    10:30/11:30/14:00/15:00 grid, so 1h is aggregated from 30m bars and 2h from
    60m bars (both half-days are exact sums of Yahoo bars);
  * intraday prices are not dividend-adjusted; the daily Adj Close / Close
    ratio is applied to every scale so all scales share one adjustment.

Yahoo only serves limited intraday history (1m: ~30 days, 5m/15m/30m: ~60
days, 60m: ~730 days). By default an existing output file is merged with the
new download, so running the script regularly accumulates history.

Usage:
    uv run --extra yahoo python scripts/download_yahoo_ashare.py
    uv run --extra yahoo python scripts/download_yahoo_ashare.py --tickers 600031.SH,300059.SZ --scales 1d,30m
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit_v1.data.calendar import SESSION_TEMPLATES, _daily_grid_minutes  # noqa: E402

MARKET = "ashare"
TZ = "Asia/Shanghai"
SCALES = ["1m", "5m", "15m", "30m", "1h", "2h", "1d"]
PRICE_COLS = ["open", "high", "low", "close"]
COLS = ["datetime", *PRICE_COLS, "volume"]

# scale -> Yahoo interval it is built from
SOURCE_INTERVAL: Dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "30m", "2h": "60m", "1d": "1d",
}
# Yahoo interval -> (how far back it is served, max days per request); None = unlimited
INTERVAL_LIMITS: Dict[str, tuple] = {
    "1m": (29, 7), "5m": (59, 59), "15m": (59, 59), "30m": (59, 59),
    "60m": (729, 729), "1d": (None, None),
}
# A bar is kept only once it closed this long ago (Yahoo's A-share feed is delayed).
SETTLE = pd.Timedelta(minutes=30)


def yahoo_symbol(code: str) -> str:
    """600031.SH -> 600031.SS; 000001.SZ stays as is."""
    num, _, exch = code.partition(".")
    exch = exch.upper()
    if exch == "SH":
        return f"{num}.SS"
    if exch == "SZ":
        return f"{num}.SZ"
    raise ValueError(f"unsupported exchange suffix in {code!r} (expected .SH or .SZ)")


def _history(ticker, retries: int, pause: float, **kw) -> pd.DataFrame:
    """One Yahoo request with retries; returns an empty frame if nothing comes back."""
    err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = ticker.history(auto_adjust=False, actions=False, **kw)
            time.sleep(pause)
            if df is not None and len(df):
                return df
            return pd.DataFrame()
        except Exception as e:  # noqa: BLE001  (network errors, rate limits)
            err = e
            time.sleep(pause + 2.0 * (attempt + 1))
    print(f"    [warn] request failed after {retries} tries ({kw}): {err}")
    return pd.DataFrame()


def fetch_interval(ticker, interval: str, now: pd.Timestamp, retries: int,
                   pause: float) -> pd.DataFrame:
    """Everything Yahoo serves for one interval, as a naive-Shanghai frame indexed
    by bar OPEN time with columns open/high/low/close/volume (+ adj_close for 1d)."""
    back, chunk = INTERVAL_LIMITS[interval]
    if back is None:
        parts = [_history(ticker, retries, pause, interval=interval, period="max")]
    else:
        parts = []
        end = now.normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=back)
        lo = start
        while lo < end:
            hi = min(lo + pd.Timedelta(days=chunk), end)
            parts.append(_history(ticker, retries, pause, interval=interval,
                                  start=lo.tz_localize(TZ), end=hi.tz_localize(TZ)))
            lo = hi
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(TZ).tz_localize(None)
    df.index = idx
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Volume": "volume", "Adj Close": "adj_close"})
    keep = [c for c in [*PRICE_COLS, "volume", "adj_close"] if c in df.columns]
    df = df[keep].astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=PRICE_COLS)


def close_stamps(open_idx: pd.DatetimeIndex, scale: str, interval: str) -> pd.DatetimeIndex:
    """Close timestamp, on the session grid of `scale`, of the bar each Yahoo bar belongs to."""
    days = open_idx.normalize()
    if scale == "1d":
        return days + pd.Timedelta(minutes=SESSION_TEMPLATES[MARKET][-1][1])
    (s1, e1), (s2, e2) = SESSION_TEMPLATES[MARKET]
    grid = np.asarray(_daily_grid_minutes(MARKET, scale), dtype=np.int64)
    m = (open_idx.hour * 60 + open_idx.minute).to_numpy().astype(np.int64)
    m = np.where(m < s1, s1, m)
    lunch = (m >= e1) & (m < s2)
    # Yahoo's 60m bar stamped 12:30 holds 13:00-13:30; any other bar opening at or
    # after 11:30 is a stub of the morning session.
    m = np.where(lunch, s2 if interval == "60m" else e1 - 1, m)
    m = np.where(m >= e2, e2 - 1, m)
    close_min = grid[np.searchsorted(grid, m, side="right")]
    return days + pd.to_timedelta(close_min, unit="m")


def build_scale(src: pd.DataFrame, scale: str, interval: str) -> pd.DataFrame:
    """Aggregate Yahoo bars into close-stamped bars of `scale` (index = close time)."""
    if src.empty:
        return pd.DataFrame(columns=[*PRICE_COLS, "volume"])
    key = close_stamps(src.index, scale, interval)
    out = src.groupby(key, sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"))
    # Yahoo occasionally serves a high/low that does not contain open/close
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    out.index.name = "datetime"
    return out


def fill_grid(bars: pd.DataFrame, scale: str) -> pd.DataFrame:
    """Complete every traded day to the full session grid: a missing bar is flat
    at the previous close with zero volume (as in the reference data)."""
    if bars.empty or scale == "1d":
        return bars
    grid = pd.to_timedelta(_daily_grid_minutes(MARKET, scale), unit="m")
    days = bars.index.normalize().unique()
    full = pd.DatetimeIndex(np.add.outer(days.values, grid.values).ravel(), name="datetime")
    out = bars.reindex(full)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out.dropna(subset=["close"])   # bars before the first real one


def adjustment_factor(daily: pd.DataFrame, mode: str) -> Optional[pd.Series]:
    """Per-day price multiplier from the daily Adj Close / Close ratio.

    back:    anchored at the first day Yahoo has (prices grow with dividends,
             like the hfq reference data);
    forward: Yahoo's own convention (latest prices are the real ones);
    none:    no adjustment.
    """
    if mode == "none" or daily.empty or "adj_close" not in daily.columns:
        return None
    f = (daily["adj_close"] / daily["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    f = f[f > 0]
    if f.empty:
        return None
    f.index = f.index.normalize()
    return f / f.iloc[0] if mode == "back" else f


def apply_adjustment(bars: pd.DataFrame, factor: Optional[pd.Series]) -> pd.DataFrame:
    if factor is None or bars.empty:
        return bars
    days = bars.index.normalize()
    # days missing from the daily table (e.g. today) take the nearest known factor
    f = factor.reindex(factor.index.union(days.unique())).ffill().bfill().reindex(days).to_numpy()
    out = bars.copy()
    for c in PRICE_COLS:
        out[c] = out[c].to_numpy() * f
    return out


def merge_with_existing(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Keep older bars already on disk; bars present in both come from the new download.

    The old part is rescaled by the median close ratio on the overlap, so a
    dividend or split since the last run cannot leave a step in the series.
    """
    if not path.is_file():
        return new
    old = pd.read_parquet(path)
    if old.empty or not set(COLS) <= set(old.columns):
        return new
    old = old[COLS].set_index("datetime").sort_index()
    if new.empty:
        return old
    common = old.index.intersection(new.index)
    if len(common):
        ratio = float(np.nanmedian(new.loc[common, "close"] / old.loc[common, "close"]))
        if np.isfinite(ratio) and ratio > 0 and abs(ratio - 1.0) > 1e-6:
            old[PRICE_COLS] = old[PRICE_COLS] * ratio
    else:
        print(f"    [warn] {path.name}: no overlap with the existing file, "
              f"appending without rescaling (gap {old.index[-1]} -> {new.index[0]})")
    older = old[~old.index.isin(new.index)]
    return pd.concat([older, new]).sort_index()


def write_parquet(bars: pd.DataFrame, path: Path) -> None:
    df = bars.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"]).astype("datetime64[ns]")
    df = df[COLS].astype({c: "float64" for c in [*PRICE_COLS, "volume"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_trade_dates(out: Path, days: set, sources: List[Path]) -> None:
    """trade_dates.csv = known calendars + every day Yahoo returned a daily bar for."""
    all_days = set(days)
    for src in sources:
        if src.is_file():
            col = pd.read_csv(src, encoding="utf-8-sig").iloc[:, 0].astype(str)
            all_days |= set(pd.to_datetime(col, format="%Y%m%d", errors="coerce").dropna())
    if not all_days:
        return
    ser = pd.Series(sorted(pd.Timestamp(d).strftime("%Y%m%d") for d in all_days), name="trade_date")
    ser.to_csv(out / "trade_dates.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruments", type=Path, default=ROOT / "data" / "instruments.json",
                    help="ticker list (keys of this JSON are used)")
    ap.add_argument("--out", type=Path, default=ROOT / "data_yahoo",
                    help="output data root (same layout as data/); pass data/ to overwrite the demo data")
    ap.add_argument("--scales", default=",".join(SCALES))
    ap.add_argument("--tickers", default="", help="comma-separated subset, e.g. 600031.SH,300059.SZ")
    ap.add_argument("--max-inst", type=int, default=0, help="only the first N tickers, 0 = all")
    ap.add_argument("--start", default="2016-08-01", help="first day kept for 1d bars")
    ap.add_argument("--adjust", choices=("back", "forward", "none"), default="back",
                    help="price adjustment; 'back' mirrors the reference data")
    ap.add_argument("--fill", action=argparse.BooleanOptionalAction, default=True,
                    help="fill missing intraday bars of traded days (flat price, zero volume)")
    ap.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True,
                    help="merge with existing output files instead of replacing them")
    ap.add_argument("--pause", type=float, default=0.4, help="seconds between Yahoo requests")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance is missing: run with `uv run --extra yahoo python ...` "
                         "or `pip install yfinance`")
    # empty windows (weekends, holidays) are expected; failures are reported by _history
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    bad = [s for s in scales if s not in SCALES]
    if bad:
        raise SystemExit(f"unknown scales {bad}; choose from {SCALES}")
    instruments = json.loads(args.instruments.read_text(encoding="utf-8"))
    codes = list(instruments)
    if args.tickers:
        want = [t.strip() for t in args.tickers.split(",") if t.strip()]
        missing = [t for t in want if t not in instruments]
        if missing:
            raise SystemExit(f"not in {args.instruments}: {missing}")
        codes = want
    if args.max_inst > 0:
        codes = codes[:args.max_inst]

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    src_root = args.instruments.resolve().parent
    for name in ("instruments.json", "stats.json"):
        src, dst = src_root / name, out / name
        if src.is_file() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)

    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    cutoff = now - SETTLE
    start = pd.Timestamp(args.start)
    intervals = sorted({SOURCE_INTERVAL[s] for s in scales} | {"1d"})   # 1d also gives the factor
    trade_days: set = set()
    failed: List[str] = []
    counts: Dict[str, List[int]] = {s: [] for s in scales}

    print(f"{len(codes)} tickers -> {out}  scales={scales}  adjust={args.adjust}")
    for i, code in enumerate(codes, 1):
        t0 = time.time()
        ticker = yf.Ticker(yahoo_symbol(code))
        raw = {iv: fetch_interval(ticker, iv, now, args.retries, args.pause) for iv in intervals}
        daily = raw["1d"]
        if daily.empty:
            print(f"[{i}/{len(codes)}] {code}: no daily data from Yahoo, skipped")
            failed.append(code)
            continue
        factor = adjustment_factor(daily, args.adjust)
        trade_days |= set(daily.index.normalize()[daily.index.normalize() >= start])
        summary = []
        for scale in scales:
            iv = SOURCE_INTERVAL[scale]
            bars = build_scale(raw[iv][[*PRICE_COLS, "volume"]] if len(raw[iv]) else raw[iv], scale, iv)
            bars = bars[bars.index <= cutoff]          # drop bars still forming
            if scale == "1d":
                bars = bars[bars.index >= start]
            if args.fill:
                bars = fill_grid(bars, scale)
            bars = apply_adjustment(bars, factor)
            path = out / MARKET / scale / f"{code}.parquet"
            if args.merge:
                bars = merge_with_existing(bars, path)
            if bars.empty:
                summary.append(f"{scale}:0")
                continue
            write_parquet(bars, path)
            counts[scale].append(len(bars))
            summary.append(f"{scale}:{len(bars)}")
        print(f"[{i}/{len(codes)}] {code}  {'  '.join(summary)}  ({time.time() - t0:.1f}s)")

    if "1d" in scales:
        write_trade_dates(out, trade_days, [out / "trade_dates.csv", src_root / "trade_dates.csv"])

    print("\nbars per ticker (min / median):")
    for scale in scales:
        c = counts[scale]
        print(f"  {scale:>3}: " + (f"{min(c)} / {int(np.median(c))}  ({len(c)} files)" if c else "no data"))
    if failed:
        print(f"\nno data for {len(failed)} tickers: {', '.join(failed)}")


if __name__ == "__main__":
    main()
