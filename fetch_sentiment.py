# fetch_sentiment.py
import math
import re
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict
from snownlp import SnowNLP
from config import ALPHA_SENTIMENT_MEAN, BETA_NEWS_VOLUME_LOG, KEYWORDS

# 简单 RSS 源（占位，能跑；后续可换成更稳定的数据源）
RSS_SOURCES = [
    "https://news.google.com/rss/search?q=%E5%B7%A5%E5%95%86%E9%93%B6%E8%A1%8C%20OR%20%E5%B7%A5%E8%A1%8C%20ICBC&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    "https://news.google.com/rss/search?q=Industrial%20and%20Commercial%20Bank%20of%20China&hl=en-US&gl=US&ceid=US:en",
]

def _clean_text(t: str) -> str:
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def fetch_news_items(max_items: int = 300) -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    items = []
    for url in RSS_SOURCES:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            entries = re.findall(r"<item>(.*?)</item>", r.text, re.S)
            for e in entries:
                title_m = re.search(r"<title>(.*?)</title>", e, re.S)
                title = _clean_text(title_m.group(1)) if title_m else ""
                date_m = re.search(r"<pubDate>(.*?)</pubDate>", e)
                pubDate = _clean_text(date_m.group(1)) if date_m else ""
                if any(k in title for k in KEYWORDS):
                    items.append({"title": title, "pubDate": pubDate})
        except Exception:
            continue
    return items[:max_items]

def score_sentiment_zh(text: str) -> float:
    try:
        return float(SnowNLP(text).sentiments)  # (0,1)
    except Exception:
        return 0.5

def daily_sentiment_aggregate(news: list[dict]) -> pd.DataFrame:
    by_day = defaultdict(list)
    for it in news:
        try:
            d = pd.to_datetime(it["pubDate"]).date()
        except Exception:
            d = datetime.utcnow().date()
        by_day[d].append(it)

    rows = []
    for d, items in by_day.items():
        scores = [score_sentiment_zh(i["title"]) for i in items if i.get("title")]
        if not scores:
            continue
        mean_score = sum(scores) / len(scores)
        volume = len(scores)
        senti_index = (ALPHA_SENTIMENT_MEAN * mean_score) + (BETA_NEWS_VOLUME_LOG * math.log(volume + 1))
        rows.append({"ds": pd.to_datetime(d), "sentiment_mean": mean_score, "volume": volume, "sentiment_index": senti_index})

    if not rows:
        return pd.DataFrame(columns=["ds", "sentiment_mean", "volume", "sentiment_index"])

    df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    return df

def build_sentiment_index_csv(save_path: str) -> pd.DataFrame:
    try:
        news = fetch_news_items()
        df = daily_sentiment_aggregate(news)
    except Exception:
        df = pd.DataFrame(columns=["ds", "sentiment_mean", "volume", "sentiment_index"])
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df
