# ensemble.py
import numpy as np
import pandas as pd
from config import W_TIMESFM, W_SENTI, W_TECH

def combine_signals(
    price_df: pd.DataFrame,             # 历史价格（含 ds, Close）
    tfm_df: pd.DataFrame,               # TimesFM 预测（含 ds, timesfm）
    senti_df: pd.DataFrame,             # 舆情按日（含 ds, sentiment_index）
    tech_score: pd.Series               # 与 price_df 对齐的技术分（-1~+1）
) -> dict:
    """
    输出：{
       'exp_return': 预测未来N天均值涨跌幅 (TimesFM),
       'senti_score': 最近一日舆情指数（标准化到0~1或-1~+1）,
       'tech_score': 最近一日技术分（-1~+1）,
       'combo_score': 综合分（-1~+1）
    }
    """
    # 1) TimesFM 预期收益：用最后一个已知Close与未来均值比较
    last_close = price_df["Close"].iloc[-1]
    exp_mean = tfm_df["timesfm"].mean()
    exp_return = (exp_mean - last_close) / (last_close + 1e-9)  # 预期平均收益率

    # 2) 舆情：取最近一天（没有就0.5/0）
    if len(senti_df):
        last_senti = senti_df.sort_values("ds")["sentiment_index"].iloc[-1]
        # 简单缩放：将均值+log量的指数，映射到 -1~+1 近似
        # 经验做法：减去历史均值再除以std并tanh
        mu = senti_df["sentiment_index"].mean()
        sd = senti_df["sentiment_index"].std() or 1.0
        senti_score = float(np.tanh((last_senti - mu) / (sd)))
    else:
        senti_score = 0.0

    # 3) 技术面：取最后一个分
    tech_last = float(tech_score.iloc[-1]) if len(tech_score) else 0.0

    # 4) 归一化 TimesFM 信号到 -1~+1（假设 ±5% 对应 ±1，超出截断）
    tfm_signal = float(np.clip(exp_return / 0.05, -1, 1))

    combo = W_TIMESFM * tfm_signal + W_SENTI * senti_score + W_TECH * tech_last
    combo = float(np.clip(combo, -1, 1))

    return {
        "exp_return": float(exp_return),
        "tfm_signal": tfm_signal,
        "senti_score": senti_score,
        "tech_score": tech_last,
        "combo_score": combo
    }
