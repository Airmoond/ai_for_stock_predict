# fetch_price.py
import os
import pandas as pd
import yfinance as yf
import akshare as ak

def fetch_price_to_csv(symbol: str, start: str, end: str | None, save_path: str) -> pd.DataFrame:
    if os.path.exists(save_path):
        print(f"[INFO] 发现本地缓存 {save_path}，直接加载")
        df = pd.read_csv(save_path, parse_dates=["ds"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df

    df = None

    if symbol.endswith(".SS"):
        # A股 → akshare (用 stock_zh_a_daily)
        ts_code = "sh" + symbol.replace(".SS", "")  # "601398.SS" -> "sh601398"
        print(f"[INFO] 尝试用 akshare 抓取 A股数据: {ts_code}")
        try:
            df = ak.stock_zh_a_daily(symbol=ts_code, adjust="qfq")
        except Exception as e:
            print(f"[ERROR] akshare 抓取失败: {e}")
            df = pd.DataFrame()

        if df is not None and not df.empty:
            df = df.reset_index().rename(columns={"date": "ds", "close": "Close"})
            df["ds"] = pd.to_datetime(df["ds"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.sort_values("ds").reset_index(drop=True)
        else:
            raise ValueError(f"akshare 下载 A股 {ts_code} 数据失败")

    else:
        # 港股/美股 → yfinance
        print(f"[INFO] 使用 yfinance 抓取数据: {symbol}")
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"yfinance 下载 {symbol} 数据失败")
        df = df.reset_index().rename(columns={"Date": "ds"})
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 数据已保存到 {save_path}，共 {len(df)} 行")
    return df
