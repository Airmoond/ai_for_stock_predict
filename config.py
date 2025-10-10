# config.py
from datetime import datetime

# Yahoo Finance 映射（保持你之前的习惯）
YF_SYMBOLS = {
    "ICBC_A": "601398.SS",  # 工商银行 A 股
    "ICBC_H": "1398.HK",    # 工商银行 港股
}

# 行情时间范围
PRICE_START_DATE = "2020-01-01"
PRICE_END_DATE = None  # 到今天

# 舆情关键词
KEYWORDS = ["工商银行", "工行", "ICBC", "Industrial and Commercial Bank of China"]

# 舆情指数参数（先行规则）
ALPHA_SENTIMENT_MEAN = 0.7
BETA_NEWS_VOLUME_LOG = 0.3

# TimesFM 预测地平线（天）
FORECAST_HORIZON = 3

# 技术面阈值
RSI_OVERBOUGHT = 70
RSI_OVERSOLD  = 30

# 融合权重（MVP，可后续学习替换）
W_TIMESFM = 0.55
W_SENTI   = 0.25
W_TECH    = 0.20

# 建议阈值
BUY_TH  = +0.30
HOLD_TH = +0.10
CUT_TH  = -0.30

# 输出目录
REPORT_DIR = "reports"
