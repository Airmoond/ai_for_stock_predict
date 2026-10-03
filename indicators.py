"""Technical signals with neutral ties and correct RSI edge cases."""
import numpy as np
import pandas as pd
from config import RSI_OVERBOUGHT, RSI_OVERSOLD, VOLUME_SURGE_RATIO


def sma(series, window):
    return series.rolling(window, min_periods=window).mean()


def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    result = 100 - 100 / (1 + up / down.replace(0, np.nan))
    result = result.mask((down == 0) & (up > 0), 100)
    result = result.mask((up == 0) & (down > 0), 0)
    return result.fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return line, signal_line, line - signal_line


def technical_score(df):
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    short, long = sma(close, 10), sma(close, 30)
    strength = rsi(close)
    _, _, hist = macd(close)
    trend = np.where(short > long, 0.4, np.where(short < long, -0.4, 0))
    reversal = np.where(strength < RSI_OVERSOLD, 0.3, np.where(strength > RSI_OVERBOUGHT, -0.3, 0))
    momentum = np.where(hist > 0, 0.3, np.where(hist < 0, -0.3, 0))
    score = np.clip(trend + reversal + momentum, -1, 1)
    score = np.where(long.notna(), score, 0)
    return pd.Series(score, index=df.index, name="tech_score")


def technical_snapshot(df):
    close = pd.to_numeric(df.Close)
    _, _, hist = macd(close)
    long = sma(close, 30).iloc[-1]
    strength = float(rsi(close).iloc[-1])
    volume_ratio = None
    if "volume" in df and len(df) >= 21:
        volume = pd.to_numeric(df.volume, errors="coerce")
        baseline = float(volume.iloc[-21:-1].mean())
        if np.isfinite(baseline) and baseline > 0 and np.isfinite(volume.iloc[-1]):
            volume_ratio = float(volume.iloc[-1] / baseline)
    cross = len(hist) >= 2 and hist.iloc[-2] <= 0 < hist.iloc[-1]
    return {"sma30": float(long) if np.isfinite(long) else None,
            "rsi14": strength, "macd_hist": float(hist.iloc[-1]),
            "volume_ratio": volume_ratio, "macd_golden_cross": bool(cross),
            "below_sma30_and_oversold": bool(np.isfinite(long) and close.iloc[-1] < long and strength < RSI_OVERSOLD),
            "volume_surge_and_golden_cross": bool(cross and volume_ratio is not None and volume_ratio >= VOLUME_SURGE_RATIO)}
