"""Combine finite signals; exclude future, outdated and legacy sentiment values."""
import numpy as np
import pandas as pd
from config import W_TIMESFM, W_SENTI, W_TECH, SENTIMENT_LOOKBACK_DAYS, SENTIMENT_MAX_AGE_DAYS
from fetch_sentiment import normalize_sentiment


def recent_sentiment(senti_df, as_of):
    if senti_df is None or senti_df.empty:
        return pd.DataFrame(columns=["ds", "sentiment_index"])
    df = normalize_sentiment(senti_df)
    return df[(df.ds <= as_of) & (df.ds >= as_of - pd.Timedelta(days=SENTIMENT_LOOKBACK_DAYS - 1))]


def sentiment_details(senti_df, as_of):
    df = recent_sentiment(senti_df, as_of)
    result = {"score": 0.0, "available": False, "as_of": None, "shock": False, "shock_evaluable": False}
    if df.empty:
        return result
    latest = df.iloc[-1]
    age = (as_of - latest.ds).days
    result["as_of"] = str(latest.ds.date())
    if age > SENTIMENT_MAX_AGE_DAYS:
        return result
    result.update(score=float(latest.sentiment_index), available=True)
    # Compare the latest observed change against earlier daily changes, excluding itself.
    changes = df.set_index("ds").sentiment_index.diff().dropna()
    if len(changes) >= 6:
        previous = changes.iloc[:-1]
        sd = float(previous.std())
        result["shock_evaluable"] = bool(np.isfinite(sd) and sd > 1e-8
                                          and (df.ds.iloc[-1] - df.ds.iloc[-2]).days == 1)
        # A shock must also be negative in absolute terms.
        result["shock"] = bool(np.isfinite(sd) and sd > 1e-8 and changes.iloc[-1] < 0
                               and changes.iloc[-1] < float(previous.mean()) - 2 * sd
                               and (df.ds.iloc[-1] - df.ds.iloc[-2]).days == 1)
    return result


def combine_signals(price_df, tfm_df, senti_df, tech_score):
    if price_df.empty or tfm_df.empty:
        raise ValueError("行情或预测为空")
    last_close = float(price_df.Close.iloc[-1])
    values = pd.to_numeric(tfm_df.timesfm, errors="coerce").to_numpy(dtype=float)
    if last_close <= 0 or not np.isfinite(last_close) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("价格或预测包含非有限值或非正值")
    as_of = pd.Timestamp(price_df.ds.iloc[-1]).normalize()
    dates = pd.to_datetime(tfm_df.ds)
    if dates.duplicated().any() or not (dates > as_of).all():
        raise ValueError("预测日期必须是行情之后的不同日期")
    exp_return = (float(values.mean()) - last_close) / last_close
    sentiment = sentiment_details(senti_df, as_of)
    tech_last = float(tech_score.iloc[-1]) if tech_score is not None and len(tech_score) else 0.0
    if not np.isfinite(tech_last):
        raise ValueError("技术面评分不是有限值")
    tfm_signal = float(np.clip(exp_return / 0.05, -1, 1))
    combo = float(np.clip(W_TIMESFM * tfm_signal + W_SENTI * sentiment["score"] + W_TECH * tech_last, -1, 1))
    return {"exp_return": exp_return, "tfm_signal": tfm_signal, "senti_score": sentiment["score"],
            "tech_score": tech_last, "combo_score": combo,
            "sentiment_available": sentiment["available"], "sentiment_as_of": sentiment["as_of"],
            "sentiment_shock": sentiment["shock"], "sentiment_shock_evaluable": sentiment["shock_evaluable"]}
