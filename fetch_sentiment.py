# fetch_sentiment.py
import re
import math
import time
import pandas as pd
import requests
from datetime import datetime
from collections import defaultdict
from snownlp import SnowNLP

# 你可以先用几个公开RSS/简单新闻源（示例用必应新闻RSS占位，建议换成你学校/实验允许的源）
RSS_SOURCES = [
    # 你可以放入若干财经RSS，或你已有的抓取接口
    "https://news.google.com/rss/search?q=%E5%B7%A5%E5%95%86%E9%93%B6%E8%A1%8C%20OR%20%E5%B7%A5%E8%A1%8C%20ICBC&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
]

def _clean_text(t: str) -> str:
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def fetch_news_items(keywords: list[str], max_items: int = 200) -> list[dict]:
    items = []
    for url in RSS_SOURCES:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            # 非严格RSS解析的简单提取：真实项目用 feedparser
            entries = re.findall(r"<item>(.*?)</item>", r.text, re.S)
            for e in entries:
                title = _clean_text(re.search(r"<title>(.*?)</title>", e, re.S).group(1))
                pubDate_m = re.search(r"<pubDate>(.*?)</pubDate>", e)
                pubDate = _clean_text(pubDate_m.group(1)) if pubDate_m else ""
                link_m = re.search(r"<link>(.*?)</link>", e)
                link = _clean_text(link_m.group(1)) if link_m else ""
                if any(k in title for k in keywords):
                    items.append({"title": title, "pubDate": pubDate, "link": link})
        except Exception:
            continue
    return items[:max_items]

def score_sentiment_zh(text: str) -> float:
    """
    SnowNLP 情感分 ∈ (0,1)，>0.5 正面
    """
    try:
        return float(SnowNLP(text).sentiments)
    except Exception:
        return 0.5

def daily_sentiment_aggregate(news: list[dict]) -> pd.DataFrame:
    by_day = defaultdict(list)
    for it in news:
        # 将pubDate解析为日期
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
        # 舆情指数：均值 + 量（对数）
        senti_index = 0.7 * mean_score + 0.3 * math.log(volume + 1)
        rows.append({"ds": pd.to_datetime(d), "sentiment_mean": mean_score, "volume": volume, "sentiment_index": senti_index})
    df = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    return df

def build_sentiment_index_csv(keywords: list[str], save_path: str) -> pd.DataFrame:
    news = fetch_news_items(keywords)
    df = daily_sentiment_aggregate(news)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df
