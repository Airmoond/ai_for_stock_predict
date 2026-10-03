"""Reusable TimesFM inference; naive baseline requires an explicit model choice."""
from functools import lru_cache
import sys
import numpy as np
import pandas as pd
from config import FORECAST_HORIZON, TIMESFM_CONTEXT, TIMESFM_CHECKPOINT
from trading_calendar import next_sessions


@lru_cache(maxsize=4)
def _get_model(output_length, backend):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("TimesFM 1.3.0 的 PyTorch 依赖要求 Python 3.11；请使用 3.11 环境和 requirements-timesfm.txt，或显式选择 --model naive")
    try:
        import timesfm
    except ImportError as exc:
        raise RuntimeError("缺少 TimesFM/PyTorch，请安装 requirements-timesfm.txt") from exc
    if not all(hasattr(timesfm, name) for name in ("TimesFm", "TimesFmHparams", "TimesFmCheckpoint")):
        raise RuntimeError("TimesFM API 不兼容，请安装 timesfm[torch]==1.3.0")
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend=backend, per_core_batch_size=1,
            horizon_len=output_length, context_len=TIMESFM_CONTEXT,
            input_patch_len=32, output_patch_len=128, num_layers=20, model_dims=1280,
            point_forecast_mode="mean"),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=TIMESFM_CHECKPOINT))


def forecast_prices(df_price, horizon=FORECAST_HORIZON, *, model="timesfm", backend="cpu"):
    if horizon < 1 or model not in ("timesfm", "naive") or backend not in ("cpu", "gpu"):
        raise ValueError("模型、设备或预测长度不合法")
    close = pd.to_numeric(df_price.Close, errors="coerce").to_numpy(dtype=float)[-TIMESFM_CONTEXT:]
    if len(close) < 30 or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("预测需要至少 30 条有效、正值的收盘价")
    if model == "naive":
        values = np.repeat(close[-1], horizon)
    else:
        output_length = max(128, ((horizon + 127) // 128) * 128)
        tfm = _get_model(output_length, backend)
        # Inputs are consecutive trading observations, independent of date gaps.
        point, _ = tfm.forecast(inputs=[close], freq=[0])
        values = np.asarray(point, dtype=float)[0, :horizon]
    if len(values) != horizon or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("模型返回无效价格或不足的预测长度")
    return values


def predict_next(df_price, horizon=FORECAST_HORIZON, *, symbol="601398.SS", model="timesfm", backend="cpu"):
    dates = next_sessions(symbol, df_price.ds.iloc[-1], horizon)
    values = forecast_prices(df_price, horizon, model=model, backend=backend)
    df = pd.DataFrame({"ds": dates, "timesfm": values})
    df.attrs["model"] = model
    return df


def predict_timesfm_next(df_price, horizon=FORECAST_HORIZON, *, symbol="601398.SS", backend="cpu"):
    return predict_next(df_price, horizon, symbol=symbol, model="timesfm", backend=backend)
