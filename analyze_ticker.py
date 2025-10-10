# analyze_ticker.py
import os
import json
import pandas as pd
from datetime import datetime
from config import *
from fetch_price import fetch_price_to_csv
from fetch_sentiment import build_sentiment_index_csv
from indicators import technical_score
from timesfm_model import predict_timesfm_next
from ensemble import combine_signals
from recommend import make_recommendation

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def analyze(ticker_alias: str, yf_symbol: str):
    ensure_dir(REPORT_DIR)

    # 1) 价格
    price_csv = os.path.join(REPORT_DIR, f"{ticker_alias}_price.csv")
    price_df = fetch_price_to_csv(yf_symbol, PRICE_START_DATE, PRICE_END_DATE, price_csv)
    price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce") 
    price_df = price_df.dropna(subset=["Close"])

    # 2) 舆情
    senti_csv = os.path.join(REPORT_DIR, f"{ticker_alias}_sentiment.csv")
    senti_df = build_sentiment_index_csv(KEYWORDS, senti_csv)

    # 3) 技术面
    tech_s = technical_score(price_df)

    # 4) TimesFM 预测
    tfm_df = predict_timesfm_next(price_df[["ds","Close"]], horizon=FORECAST_HORIZON)

    # 5) 融合与建议
    metrics = combine_signals(price_df, tfm_df, senti_df, tech_s)
    rec = make_recommendation(metrics)

    report = {
        "ticker": ticker_alias,
        "yf_symbol": yf_symbol,
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "recommendation": rec,
        "peek_forecast": tfm_df.tail(FORECAST_HORIZON).to_dict(orient="records")
    }

    # 保存
    report_json = os.path.join(REPORT_DIR, f"{ticker_alias}_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台友好输出
    print(f"\n=== {ticker_alias} 投资建议（{report['as_of']}）===")
    print(f"TimesFM 预期均值收益率: {metrics['exp_return']:.2%}")
    print(f"综合信号分（-1~+1）: {metrics['combo_score']:.3f}")
    print(f"→ 建议：{rec['rating']} | 理由：{rec['note']}")
    print("风险提示：", "；".join(rec["risks"]))
    print("触发条件：", "；".join(rec["triggers"]))
    print(f"报告JSON：{report_json}\n")

if __name__ == "__main__":
    # 示例1：工商银行 A 股
    analyze("ICBC_A", YF_SYMBOLS["ICBC_A"])
    # 示例2：工商银行 H 股（如需要）
    # analyze("ICBC_H", YF_SYMBOLS["ICBC_H"])
