# config.py
from datetime import datetime, timedelta

# 市场代码映射（Yahoo Finance）
YF_SYMBOLS = {
    "ICBC_A": "601398.SS",  # 工商银行 A股
    "ICBC_H": "1398.HK",    # 工商银行 港股
}

# 数据时间范围
PRICE_START_DATE = "2020-01-01"
PRICE_END_DATE = None  # 到今天

# 舆情抓取关键词
KEYWORDS = ["工商银行", "工行", "ICBC"]

# 舆情权重与指标参数
ALPHA_SENTIMENT_MEAN = 0.7
BETA_NEWS_VOLUME_LOG = 0.3

# TimesFM 预测地平线（天）
FORECAST_HORIZON = 3

# 技术指标阈值
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# 集成权重（先用规则法，后续可训练模型替换）
W_TIMESFM = 0.55
W_SENTI   = 0.25
W_TECH    = 0.20

# 决策阈值
BUY_TH   =  +0.30
HOLD_TH  =  +0.10
CUT_TH   =  -0.30

# 报告输出路径
REPORT_DIR = "reports"

# config.py
TUSHARE_TOKEN = "068a2c0125efd8fcd889eb845d187aa55dbcf27f9a472916495e40b0"
