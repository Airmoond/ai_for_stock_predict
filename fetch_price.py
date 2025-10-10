# fetch_price.py
import os
import pandas as pd
import yfinance as yf
import akshare as ak

def fetch_price_to_csv(symbol: str, start: str, end: str | None, save_path: str) -> pd.DataFrame:
    """
    symbol:
      - A股: 601398.SS → 使用 akshare.stock_zh_a_daily("sh601398")
      - 港/美股: 1398.HK / AAPL → 使用 yfinance
    """
    # 1) 读缓存（避免限流 & 加速）
    if os.path.exists(save_path):
        print(f"[INFO] 发现本地缓存 {save_path}，直接加载")
        df = pd.read_csv(save_path, parse_dates=["ds"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df

    # 2) 抓取
    if symbol.endswith(".SS"):
        ts_code = "sh" + symbol.replace(".SS", "")  # 601398.SS -> sh601398
        print(f"[INFO] 尝试用 akshare 抓取 A股数据: {ts_code}")
        try:
            df = ak.stock_zh_a_daily(symbol=ts_code, adjust="qfq")  # 前复权
        except Exception as e:
            print(f"[ERROR] akshare 抓取失败: {e}")
            df = pd.DataFrame()

        if df is None or df.empty:
            raise ValueError(f"akshare 下载 A股 {ts_code} 数据失败")

        df = df.reset_index().rename(columns={"date": "ds", "close": "Close"})
        df["ds"] = pd.to_datetime(df["ds"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.sort_values("ds").reset_index(drop=True)

        # 限定时间范围
        if start:
            df = df[df["ds"] >= pd.to_datetime(start)]
        if end:
            df = df[df["ds"] <= pd.to_datetime(end)]

    else:
        print(f"[INFO] 使用 yfinance 抓取数据: {symbol}")
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"yfinance 下载 {symbol} 数据失败")
        df = df.reset_index().rename(columns={"Date": "ds"})
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.sort_values("ds").reset_index(drop=True)

    # 3) 存缓存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 数据已保存到 {save_path}，共 {len(df)} 行")
    return df
