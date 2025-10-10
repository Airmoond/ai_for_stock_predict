# recommend.py
from config import BUY_TH, HOLD_TH, CUT_TH

def make_recommendation(metrics: dict) -> dict:
    """
    根据综合分阈值生成建议
    """
    cs = metrics["combo_score"]
    if cs >= BUY_TH:
        rating = "买入"
        note = "技术与舆情共振向上，且时序预测显示短期均值上涨。"
    elif cs >= HOLD_TH:
        rating = "谨慎增持/持有"
        note = "信号偏正，但共识不强，关注回撤与量化边界。"
    elif cs <= CUT_TH:
        rating = "回避/减仓"
        note = "多渠道信号偏负，短期下行风险较高。"
    else:
        rating = "观望"
        note = "信号分歧较大或强度不足，等待明确趋势。"

    # 风险提示（模板）
    risks = [
        "宏观政策与监管变化可能导致估值重定价",
        "突发负面舆情与黑天鹅事件",
        "市场整体风险偏好快速回落",
    ]

    triggers = [
        "若舆情指数日内大幅下挫（跌超 2σ）则下调评级",
        "若价格跌破30日均线且RSI<30，转为回避",
        "若成交量放大且MACD金叉，考虑上调评级"
    ]

    return {
        "rating": rating,
        "note": note,
        "risks": risks,
        "triggers": triggers
    }
