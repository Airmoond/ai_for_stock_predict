"""Language-aware RSS sentiment with signed volume weighting and safe cache reuse."""
import html
import logging
import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from config import (ALPHA_SENTIMENT_MEAN, BETA_NEWS_VOLUME_LOG, KEYWORDS,
                    SENTIMENT_LOOKBACK_DAYS)
from storage import write_csv

log = logging.getLogger(__name__)
RSS_SOURCES = [
    "https://news.google.com/rss/search?q=工商银行%20OR%20工行%20OR%20ICBC&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    "https://news.google.com/rss/search?q=Industrial%20and%20Commercial%20Bank%20of%20China&hl=en-US&gl=US&ceid=US:en",
]
COLUMNS = ["ds", "sentiment_mean", "volume", "sentiment_index", "schema_version"]


def _clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(text or ""))).strip()


def fetch_news_items(max_items=300):
    items, failures, successful = [], [], 0
    for url in RSS_SOURCES:
        try:
            response = requests.get(url, headers={"User-Agent": "ICBCResearch/2.0"}, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            successful += 1
            for entry in root.findall(".//item"):
                title = _clean_text(entry.findtext("title"))
                if any(keyword.casefold() in title.casefold() for keyword in KEYWORDS):
                    items.append({"title": title, "description": _clean_text(entry.findtext("description")),
                                  "pubDate": entry.findtext("pubDate"), "link": entry.findtext("link")})
        except (requests.RequestException, ET.ParseError) as exc:
            failures.append(str(exc))
            log.warning("新闻源获取失败: %s", exc)
    if successful == 0:
        raise RuntimeError("所有新闻源获取失败: " + "; ".join(failures))
    # Newest first, without inventing a date for malformed RSS entries.
    def published(item):
        try:
            return pd.to_datetime(item["pubDate"], utc=True).timestamp()
        except (TypeError, ValueError):
            return float("-inf")
    return sorted(items, key=published, reverse=True)[:max_items]


def score_sentiment_zh(text):
    from snownlp import SnowNLP
    return float(SnowNLP(text).sentiments)


@lru_cache(maxsize=1)
def _english_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def score_sentiment(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return score_sentiment_zh(text)
    return (_english_analyzer().polarity_scores(text)["compound"] + 1) / 2


def sentiment_index(mean, volume):
    # News count amplifies the sign instead of turning negative coverage positive.
    amplitude = ALPHA_SENTIMENT_MEAN + BETA_NEWS_VOLUME_LOG * min(math.log1p(volume) / math.log(301), 1)
    return float(np.clip((2 * mean - 1) * amplitude, -1, 1))


def normalize_sentiment(df):
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    if not {"ds", "sentiment_mean", "volume"}.issubset(df.columns):
        raise ValueError("舆情缓存缺少原始均分或新闻数量，无法转换旧版指数")
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in ("sentiment_mean", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ds", "sentiment_mean", "volume"])
    df = df[np.isfinite(df.sentiment_mean) & np.isfinite(df.volume)
            & df.sentiment_mean.between(0, 1) & (df.volume > 0)]
    df["sentiment_index"] = [sentiment_index(m, v) for m, v in zip(df.sentiment_mean, df.volume)]
    df["schema_version"] = 2
    return df[COLUMNS].sort_values("ds").drop_duplicates("ds", keep="last").reset_index(drop=True)


def daily_sentiment_aggregate(news, *, as_of=None, lookback_days=SENTIMENT_LOOKBACK_DAYS):
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    by_day, seen, errors = {}, set(), []
    for item in news:
        title = _clean_text(item.get("title"))
        if not title:
            continue
        try:
            published = pd.to_datetime(item.get("pubDate"), utc=True)
            if pd.isna(published):
                raise ValueError("新闻缺少发布日期")
            day = published.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            if day > as_of or day < as_of - pd.Timedelta(days=lookback_days - 1):
                continue
            key = (day, title.casefold())
            if key in seen:
                continue
            seen.add(key)
            description = _clean_text(item.get("description"))
            text = title if not description or description.startswith(title) else title + ". " + description
            score = score_sentiment(text)
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("情绪分不在有效范围内")
            by_day.setdefault(day, []).append(score)
        except (ValueError, TypeError, ImportError) as exc:
            errors.append(f"跳过新闻 {title[:40]}: {exc}")
    rows = [{"ds": day, "sentiment_mean": float(np.mean(scores)), "volume": len(scores)}
            for day, scores in by_day.items()]
    df = normalize_sentiment(pd.DataFrame(rows))
    df.attrs["warnings"] = errors
    return df


def build_sentiment_index_csv(save_path, *, as_of=None, offline=False):
    path = Path(save_path)
    warnings = []
    cached = pd.DataFrame(columns=COLUMNS)
    if path.exists():
        try:
            cached = normalize_sentiment(pd.read_csv(path))
        except (ValueError, pd.errors.ParserError) as exc:
            warnings.append(f"舆情缓存不可用: {exc}")
    df = cached
    source = "cache"
    if not offline:
        try:
            fresh = daily_sentiment_aggregate(fetch_news_items(), as_of=as_of)
            warnings.extend(fresh.attrs.get("warnings", []))
            if not fresh.empty:
                df = normalize_sentiment(fresh if cached.empty else pd.concat([cached, fresh], ignore_index=True))
                write_csv(df, path)
                source = "rss"
            else:
                warnings.append("没有有效近期新闻，保留已有舆情缓存")
        except Exception as exc:
            warnings.append(f"舆情更新失败，保留已有缓存: {exc}")
            log.warning(warnings[-1])
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        df = df[(df.ds <= cutoff) & (df.ds >= cutoff - pd.Timedelta(days=SENTIMENT_LOOKBACK_DAYS - 1))].copy()
    df.attrs["quality"] = {"source": source, "warnings": warnings}
    return df
