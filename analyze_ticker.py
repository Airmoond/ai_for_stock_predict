"""Command-line analysis with reproducible dates, model labels and data quality."""
import argparse
import logging
from pathlib import Path
import pandas as pd
from config import (YF_SYMBOLS, REPORT_DIR, PRICE_START_DATE, PRICE_END_DATE,
                    FORECAST_HORIZON, W_TIMESFM, W_SENTI, W_TECH, TIMESFM_CHECKPOINT)
from fetch_price import fetch_price_to_csv
from fetch_sentiment import build_sentiment_index_csv
from indicators import technical_score, technical_snapshot
from timesfm_model import predict_next
from ensemble import combine_signals
from recommend import make_recommendation
from storage import write_json


def analyze(ticker_alias, yf_symbol, *, start=PRICE_START_DATE, end=PRICE_END_DATE,
            horizon=FORECAST_HORIZON, offline=False, force_refresh=False,
            strict_data=False, model="timesfm", backend="cpu", output_dir=REPORT_DIR,
            cache_dir=None):
    if offline and force_refresh:
        raise ValueError("离线模式不能强制联网刷新")
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir
    price_df = fetch_price_to_csv(yf_symbol, start, end, cache_dir / f"{ticker_alias}_price.csv",
                                 offline=offline, force_refresh=force_refresh, allow_stale=not strict_data)
    quality = dict(price_df.attrs["quality"])
    quality["insufficient_history"] = len(price_df) < 30
    if strict_data and (quality["stale"] or quality["missing_sessions"]):
        raise ValueError("严格模式拒绝过期或不完整的行情: " + "; ".join(quality["warnings"]))
    data_as_of = price_df.ds.iloc[-1]
    senti_df = build_sentiment_index_csv(cache_dir / f"{ticker_alias}_sentiment.csv",
                                        as_of=data_as_of, offline=offline)
    forecast_df = predict_next(price_df[["ds", "Close"]], horizon=horizon,
                               symbol=yf_symbol, model=model, backend=backend)
    metrics = combine_signals(price_df, forecast_df, senti_df, technical_score(price_df))
    technical = technical_snapshot(price_df)
    recommendation = make_recommendation(metrics, technical=technical, data_quality=quality, model=model)
    warnings = quality["warnings"] + senti_df.attrs.get("quality", {}).get("warnings", [])
    if not metrics["sentiment_available"]:
        warnings.append("没有7日内有效舆情，舆情分按0计；未重新分配权重")
    report = {
        "schema_version": 2, "ticker": ticker_alias, "yf_symbol": yf_symbol,
        "as_of": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "data_as_of": str(data_as_of.date()), "requested_range": {"start": start, "end": end},
        "model": {"name": model, "checkpoint": TIMESFM_CHECKPOINT if model == "timesfm" else None,
                  "horizon_sessions": horizon, "context_observations": min(len(price_df), 512)},
        "data_quality": quality, "warnings": warnings,
        "weights": {"forecast": W_TIMESFM, "sentiment": W_SENTI, "technical": W_TECH},
        "metrics": metrics, "technical": technical, "recommendation": recommendation,
        "forecast": [{"ds": str(row.ds.date()), "price": float(row.timesfm)}
                     for row in forecast_df.itertuples(index=False)],
    }
    suffix = "" if model == "timesfm" else f"_{model}"
    path = output_dir / f"{ticker_alias}{suffix}_report.json"
    write_json(report, path)
    print(f"\n{ticker_alias} 分析｜行情截至 {report['data_as_of']}｜模型 {model}")
    print(f"评级：{recommendation['rating']}\n{recommendation['note']}")
    for warning in warnings:
        print(f"提示：{warning}")
    print(f"报告：{path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="工商银行价格、舆情和技术面分析")
    parser.add_argument("--ticker", choices=YF_SYMBOLS, default="ICBC_A")
    parser.add_argument("--start", default=PRICE_START_DATE)
    parser.add_argument("--end", default=PRICE_END_DATE, help="包含结束日，仅使用已收盘行情")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--model", choices=["timesfm", "naive"], default="timesfm")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--refresh", action="store_true")
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, help="缓存目录；默认与输出目录相同")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        analyze(args.ticker, YF_SYMBOLS[args.ticker], start=args.start, end=args.end,
                horizon=args.horizon, offline=args.offline, force_refresh=args.refresh,
                strict_data=args.strict_data, model=args.model, backend=args.backend,
                output_dir=args.output_dir, cache_dir=args.cache_dir)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"分析失败：{exc}\n")


if __name__ == "__main__":
    main()
