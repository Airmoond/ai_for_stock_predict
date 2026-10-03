"""Walk-forward forecast evaluation and delayed-execution long-only simulation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import REPORT_DIR, FORECAST_HORIZON
from fetch_price import normalize_prices
from indicators import technical_score, technical_snapshot
from timesfm_model import forecast_prices
from ensemble import combine_signals
from recommend import make_recommendation
from storage import write_csv, write_json


def forecast_metrics(actual, predicted, last):
    actual, predicted, last = (np.asarray(values, dtype=float) for values in (actual, predicted, last))
    nonflat = np.abs(actual - last) > 1e-10
    predicted_direction = np.where(np.abs(predicted - last) <= 1e-10, 0, np.sign(predicted - last))
    return {"mae": float(np.abs(predicted - actual).mean()),
            "rmse": float(np.sqrt(np.square(predicted - actual).mean())),
            "mape": float((np.abs((predicted - actual) / actual)).mean()),
            "direction_accuracy": float((predicted_direction[nonflat]
                                          == np.sign(actual[nonflat] - last[nonflat])).mean()) if nonflat.any() else None,
            "direction_samples": int(nonflat.sum())}


def portfolio_metrics(returns):
    returns = np.asarray(returns, dtype=float)
    wealth = np.concatenate([[1.0], np.cumprod(1 + returns)])
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {"total_return": float(wealth[-1] - 1), "max_drawdown": float(drawdown.min())}


def walk_forward(price_df, *, model="timesfm", backend="cpu", horizon=FORECAST_HORIZON,
                 min_train=128, test_sessions=120, cost_bps=10):
    if min_train < 30 or horizon < 1 or test_sessions < 1 or not 0 <= cost_bps < 10000:
        raise ValueError("回测参数不合法：训练至少30条，预测/测试长度为正数，成本为0至9999基点")
    df = normalize_prices(price_df)
    stop = min(len(df) - horizon, len(df) - 2)
    first = max(min_train - 1, stop - test_sessions)
    if first >= stop:
        raise ValueError("数据不足以同时覆盖训练、预测和延迟成交区间")
    rows, previous_position, cost = [], 0, cost_bps / 10000
    for origin in range(first, stop):
        history = df.iloc[:origin + 1]
        future = df.iloc[origin + 1:origin + 1 + horizon]
        predicted = forecast_prices(history, horizon, model=model, backend=backend)
        forecast = pd.DataFrame({"ds": future.ds.to_numpy(), "timesfm": predicted})
        metrics = combine_signals(history, forecast, None, technical_score(history))
        recommendation = make_recommendation(metrics, technical=technical_snapshot(history), model=model)
        position = int(recommendation["rating"] in ("买入", "谨慎增持/持有"))
        # Observe t's close, execute at t+1's close, then earn t+1 -> t+2 return.
        # No fill is assumed at the same close used to construct the signal.
        enter, exit_row = df.iloc[origin + 1], df.iloc[origin + 2]
        market_return = float(exit_row.Close / enter.Close - 1)
        turnover = abs(position - previous_position)
        net_return = (1 + position * market_return) * (1 - cost * turnover) - 1
        rows.append({"signal_date": str(history.ds.iloc[-1].date()),
                     "forecast_end": str(future.ds.iloc[-1].date()),
                     "execution_date": str(enter.ds.date()), "return_date": str(exit_row.ds.date()),
                     "last_close": float(history.Close.iloc[-1]),
                     "predicted_mean": float(predicted.mean()), "actual_mean": float(future.Close.mean()),
                     "combo_score": metrics["combo_score"], "rating": recommendation["rating"],
                     "position": position, "turnover": turnover,
                     "market_return": market_return, "strategy_return": net_return})
        previous_position = position
    # Liquidate the final position, and apply equal round-trip costs to buy-and-hold.
    rows[-1]["strategy_return"] = (1 + rows[-1]["strategy_return"]) * (1 - cost * previous_position) - 1
    rows[-1]["turnover"] += previous_position
    result = pd.DataFrame(rows)
    benchmark_returns = result.market_return.to_numpy(copy=True)
    benchmark_returns[0] = (1 + benchmark_returns[0]) * (1 - cost) - 1
    benchmark_returns[-1] = (1 + benchmark_returns[-1]) * (1 - cost) - 1
    actual = result.actual_mean.to_numpy()
    last = result.last_close.to_numpy()
    summary = {
        "model": model, "horizon_sessions": horizon, "samples": len(result),
        "first_signal_date": result.signal_date.iloc[0], "last_signal_date": result.signal_date.iloc[-1],
        "forecast": forecast_metrics(actual, result.predicted_mean, last),
        "naive_baseline": forecast_metrics(actual, last, last),
        "strategy": {**portfolio_metrics(result.strategy_return), "turnover": int(result.turnover.sum()),
                     "invested_sessions": int(result.position.sum())},
        "buy_and_hold": portfolio_metrics(benchmark_returns), "cost_bps_per_side": cost_bps,
        "method": {"training": "expanding history, model context capped at 512 observations",
                   "target": "mean price over the next horizon observed trading sessions",
                   "execution": "signal after t close; trade at t+1 close; accrue return through t+2 close",
                   "sentiment": "excluded: RSS aggregates are not point-in-time historical archives",
                   "weights": "same configured weights; unavailable sentiment contributes zero",
                   "limits": ["前复权历史价格可能经过事后修订", "未模拟冲击成本、涨跌停和停牌成交约束",
                              "短样本不代表长期效果；收益未年化"]}}
    return summary, result


def main():
    parser = argparse.ArgumentParser(description="按时间滚动回测，比较价格基准与延迟成交策略")
    parser.add_argument("--price-csv", type=Path, default=REPORT_DIR / "ICBC_A_price.csv")
    parser.add_argument("--model", choices=["timesfm", "naive"], default="timesfm")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--min-train", type=int, default=128)
    parser.add_argument("--test-sessions", type=int, default=120)
    parser.add_argument("--cost-bps", type=float, default=10)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR / "backtests")
    args = parser.parse_args()
    try:
        price = normalize_prices(pd.read_csv(args.price_csv))
        price = price[price.ds >= pd.Timestamp(args.start)]
        if args.end:
            price = price[price.ds <= pd.Timestamp(args.end)]
        summary, detail = walk_forward(price, model=args.model, backend=args.backend, horizon=args.horizon,
                                       min_train=args.min_train, test_sessions=args.test_sessions, cost_bps=args.cost_bps)
        prefix = f"{args.price_csv.stem}_{args.model}_backtest"
        write_csv(detail, args.output_dir / f"{prefix}.csv")
        write_json(summary, args.output_dir / f"{prefix}.json")
        print(f"回测完成：{summary['samples']} 个样本；MAE {summary['forecast']['mae']:.4f}；"
              f"策略净收益 {summary['strategy']['total_return']:.2%}；"
              f"买入持有净收益 {summary['buy_and_hold']['total_return']:.2%}")
        print(f"输出：{args.output_dir}")
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"回测失败：{exc}\n")


if __name__ == "__main__":
    main()
