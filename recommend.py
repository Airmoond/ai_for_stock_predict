# recommend.py
from config import BUY_TH, HOLD_TH, CUT_TH

def make_recommendation(metrics: dict) -> dict:
    cs = metrics["combo_score"]
    if cs >= BUY_TH:
        rating = "买入"
        note = "技术与舆情共振向上，时序预测显示短期上涨概率较高。"
    elif cs >= HOLD_TH:
        rating = "谨慎增持/持有"
        note = "信号偏正但强度一般，可小仓位试探并关注回撤。"
    elif cs <= CUT_TH:
        rating = "回避/减仓"
        note = "多渠道信号偏负，短期下行风险较高。"
    else:
        rating = "观望"
        note = "信号分歧或强度不足，等待趋势确认。"

    risks = [
        "宏观政策与监管变动引发估值重定价",
        "突发负面舆情或黑天鹅事件",
        "市场整体风险偏好快速回落",
    ]
    triggers = [
        "若舆情指数单日下挫超过 2σ，则下调评级",
        "若价格跌破30日均线且RSI<30，转为回避",
        "若成交量放大且MACD金叉，考虑上调评级",
    ]
    return {"rating": rating, "note": note, "risks": risks, "triggers": triggers}
