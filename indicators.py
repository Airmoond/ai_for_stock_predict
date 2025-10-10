# indicators.py
import pandas as pd
import numpy as np

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs = up / (down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def technical_score(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    s_sma = sma(close, 10)
    l_sma = sma(close, 30)
    r = rsi(close, 14)
    _, _, hist = macd(close)

    # 基于趋势/超买卖/动量的简单打分
    score = np.zeros(len(close), dtype=float)
    score += np.where((s_sma > l_sma), 0.4, -0.4)
    score += np.where(r < 30, +0.3, np.where(r > 70, -0.3, 0.0))
    score += np.where(hist > 0, 0.3, -0.3)
    score = np.clip(score, -1, 1)

    return pd.Series(score, index=df.index, name="tech_score")
