"""Analysis defaults; paths remain stable when run outside the project directory."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPORT_DIR = ROOT_DIR / "reports"
YF_SYMBOLS = {"ICBC_A": "601398.SS", "ICBC_H": "1398.HK"}
PRICE_START_DATE = "2020-01-01"
PRICE_END_DATE = None
PRICE_CACHE_MAX_AGE_HOURS = 24
KEYWORDS = ["工商银行", "工行", "ICBC", "Industrial and Commercial Bank of China"]
ALPHA_SENTIMENT_MEAN = 0.7
BETA_NEWS_VOLUME_LOG = 0.3
SENTIMENT_LOOKBACK_DAYS = 30
SENTIMENT_MAX_AGE_DAYS = 7
FORECAST_HORIZON = 3  # Trading sessions, not calendar days.
TIMESFM_CONTEXT = 512
TIMESFM_CHECKPOINT = "google/timesfm-1.0-200m-pytorch"
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
W_TIMESFM = 0.55
W_SENTI = 0.25
W_TECH = 0.20
BUY_TH = 0.30
HOLD_TH = 0.10
CUT_TH = -0.30
VOLUME_SURGE_RATIO = 1.5
