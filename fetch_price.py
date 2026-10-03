"""Daily adjusted prices, with range-aware refresh and explicit stale fallback."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from config import PRICE_CACHE_MAX_AGE_HOURS
from storage import write_csv
from trading_calendar import latest_completed_session, sessions_in_range

log = logging.getLogger(__name__)


def normalize_prices(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (price field, ticker) even for a single ticker.
        df.columns = df.columns.get_level_values(0)
        if df.columns.duplicated().any():
            raise ValueError("行情包含多个股票，无法归一化")
    if "ds" not in df:
        df = df.reset_index()
    df = df.rename(columns={"date": "ds", "Date": "ds", "Datetime": "ds",
                            "close": "Close", "Volume": "volume"})
    if not {"ds", "Close"}.issubset(df.columns):
        raise ValueError("行情缺少 ds 或 Close 列")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["ds", "Close"])
    df = df[np.isfinite(df["Close"]) & (df["Close"] > 0)]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    df = df.drop(columns=["index"], errors="ignore")
    return df.sort_values("ds").drop_duplicates("ds", keep="last").reset_index(drop=True)


def _download_prices(symbol, start, end):
    if symbol.endswith((".SS", ".SZ")):
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("A 股行情需要 akshare，请安装 requirements.txt") from exc
        code = ("sh" if symbol.endswith(".SS") else "sz") + symbol.split(".")[0]
        # Ex-dividend events can revise old qfq prices; replace rather than append.
        return ak.stock_zh_a_daily(symbol=code, adjust="qfq")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("港/美股行情需要 yfinance，请安装 requirements.txt") from exc
    return yf.download(symbol, start=start, end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
                       auto_adjust=True, progress=False, threads=False, timeout=15)


def fetch_price_to_csv(symbol, start, end, save_path, *, offline=False,
                       force_refresh=False, allow_stale=True, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    target = latest_completed_session(symbol, end=end, now=now)
    start_date = pd.Timestamp(start).normalize()
    if start_date > target:
        raise ValueError("起始日期晚于最后一个已收盘交易日")
    expected = sessions_in_range(symbol, start_date, target)
    path = Path(save_path)
    warnings = []
    cached = None

    def bounded(frame):
        frame = normalize_prices(frame)
        return frame[(frame.ds >= start_date) & (frame.ds <= target)].reset_index(drop=True)

    if path.exists():
        try:
            cached = bounded(pd.read_csv(path))
        except (ValueError, pd.errors.ParserError) as exc:
            warnings.append(f"行情缓存不可用: {exc}")
    complete = (cached is not None and not cached.empty
                and cached.ds.iloc[0] <= expected[0] and cached.ds.iloc[-1] >= target
                and len(expected.difference(pd.DatetimeIndex(cached.ds))) == 0)
    young = path.exists() and (now.timestamp() - path.stat().st_mtime <= PRICE_CACHE_MAX_AGE_HOURS * 3600)
    source = "cache"
    if offline:
        if cached is None or cached.empty:
            raise ValueError("离线模式没有指定范围内的行情缓存")
        df = cached
    elif complete and young and not force_refresh:
        df = cached
    else:
        try:
            df = bounded(_download_prices(symbol, start, target.date().isoformat()))
            if df.empty:
                raise ValueError("数据源返回空行情")
            write_csv(df, path)
            source = "download"
        except Exception as exc:
            if not allow_stale or cached is None or cached.empty:
                raise RuntimeError(f"行情更新失败且没有可用缓存: {exc}") from exc
            df = cached
            source = "cache_fallback"
            warnings.append(f"行情更新失败，使用缓存: {exc}")
            log.warning(warnings[-1])
    stale = bool(df.ds.iloc[-1] < target)
    missing = expected.difference(pd.DatetimeIndex(df.ds))
    if stale:
        warnings.append(f"行情截至 {df.ds.iloc[-1].date()}，应至少更新至 {target.date()}")
    if len(missing):
        warnings.append(f"指定范围内缺少 {len(missing)} 个交易日记录（可能停牌或数据缺失）")
    df.attrs["quality"] = {"source": source, "offline": offline, "stale": stale,
                           "expected_as_of": str(target.date()), "missing_sessions": len(missing),
                           "warnings": warnings}
    return df
