# timesfm_model.py
import pandas as pd
import timesfm
from config import FORECAST_HORIZON

def predict_timesfm_next(df_price: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    输入: df_price 含列 ['ds','Close']
    输出: 未来 horizon 天的 df（列: ds, timesfm）
    """
    df2 = df_price[["ds", "Close"]].rename(columns={"Close": "y"}).copy()
    df2["unique_id"] = "TICK"
    if len(df2) > 512:
        df2 = df2.iloc[-512:].copy()

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=32,
            horizon_len=horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

    forecast_df = tfm.forecast_on_df(inputs=df2, freq="D", value_name="y", num_jobs=1)
    return forecast_df[["ds", "timesfm"]]
