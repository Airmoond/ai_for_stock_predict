# ensemble.py
import numpy as np
import pandas as pd
from config import W_TIMESFM, W_SENTI, W_TECH

def combine_signals(
    price_df: pd.DataFrame,
    tfm_df: pd.DataFrame,
    senti_df: pd.DataFrame,
    tech_score: pd.Series
) -> dict:
    last_close = float(price_df["Close"].iloc[-1])

    # TimesFM 预期均值收益率
    exp_mean = float(tfm_df["timesfm"].mean())
    exp_return = (exp_mean - last_close) / (last_close + 1e-9)

    # 舆情分：标准化到 -1~+1
    if senti_df is not None and len(senti_df) > 0:
        last_senti = float(senti_df.sort_values("ds")["sentiment_index"].iloc[-1])
        mu = float(senti_df["sentiment_index"].mean())
        sd = float(senti_df["sentiment_index"].std() or 1.0)
        senti_score = float(np.tanh((last_senti - mu) / (sd)))
    else:
        senti_score = 0.0

    # 技术分（最后一个）
    tech_last = float(tech_score.iloc[-1]) if tech_score is not None and len(tech_score) > 0 else 0.0

    # TimesFM 信号归一（±5% → ±1）
    tfm_signal = float(np.clip(exp_return / 0.05, -1, 1))

    combo = float(np.clip(W_TIMESFM * tfm_signal + W_SENTI * senti_score + W_TECH * tech_last, -1, 1))

    return {
        "exp_return": exp_return,
        "tfm_signal": tfm_signal,
        "senti_score": senti_score,
        "tech_score": tech_last,
        "combo_score": combo,
    }
