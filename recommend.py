"""Explain the actual components and evaluate conditions using current observations."""
import numpy as np
from config import BUY_TH, HOLD_TH, CUT_TH


def make_recommendation(metrics, *, technical=None, data_quality=None, model="timesfm"):
    cs = float(metrics["combo_score"])
    if not np.isfinite(cs):
        raise ValueError("综合评分无效")
    if cs >= BUY_TH:
        rating = "买入"
    elif cs >= HOLD_TH:
        rating = "谨慎增持/持有"
    elif cs <= CUT_TH:
        rating = "回避/减仓"
    else:
        rating = "观望"
    technical = technical or {}
    evaluations = [
        {"condition": "舆情日变化较此前变化均值低超过 2σ",
         "evaluated": bool(metrics.get("sentiment_shock_evaluable", metrics.get("sentiment_shock", False))),
         "triggered": bool(metrics.get("sentiment_shock")), "action": "下调一级评级"},
        {"condition": "收盘价低于30日均线且RSI低于30",
         "evaluated": technical.get("sma30") is not None,
         "triggered": bool(technical.get("below_sma30_and_oversold")), "action": "转为回避/减仓"},
        {"condition": "成交量至少为此前20日均量的1.5倍且MACD金叉",
         "evaluated": technical.get("volume_ratio") is not None,
         "triggered": bool(technical.get("volume_surge_and_golden_cross")), "action": "关注增强信号，仍按综合分评级"},
    ]
    if evaluations[0]["triggered"]:
        rating = {"买入": "谨慎增持/持有", "谨慎增持/持有": "观望",
                  "观望": "回避/减仓", "回避/减仓": "回避/减仓"}[rating]
    if evaluations[1]["triggered"]:
        rating = "回避/减仓"
    quality = data_quality or {}
    if quality.get("stale") or quality.get("missing_sessions", 0) or quality.get("insufficient_history"):
        rating = "数据不足"
    source = "TimesFM" if model == "timesfm" else "最后收盘价基准（非 AI）"
    sentiment = f"{metrics['senti_score']:+.3f}" if metrics.get("sentiment_available") else "近期数据缺失，按0计分"
    note = (f"{source}预测均值相对最后收盘价变化 {metrics['exp_return']:+.2%}；"
            f"舆情分 {sentiment}；技术分 {metrics['tech_score']:+.3f}；综合分 {cs:+.3f}。")
    if rating == "数据不足":
        note += "行情过期或交易记录不完整，暂停给出买卖评级。"
    return {"rating": rating, "note": note,
            "risks": ["宏观政策与监管变化", "突发事件", "市场整体波动"],
            "triggers": [f"{item['condition']}：{'数据不足，无法评估' if not item['evaluated'] else ('已触发' if item['triggered'] else '未触发')}"
                         for item in evaluations],
            "trigger_evaluations": evaluations}
