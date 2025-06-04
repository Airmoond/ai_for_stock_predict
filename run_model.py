import pandas as pd
import timesfm
import torch

# 读取 CSV 并预处理
df = pd.read_csv("aapl_stock.csv", parse_dates=["Date"])
df2 = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
df2["unique_id"] = "AAPL"

# 限制最近512条数据
if len(df2) > 512:
    df2 = df2.iloc[-512:].copy()

# 加载模型
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="torch",
        per_core_batch_size=32,
        horizon_len=1,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

# 预测
forecast_df = tfm.forecast_on_df(
    inputs=df2,
    freq="D",
    value_name="y",
    num_jobs=1
)

# 正确使用预测列："timesfm"
predicted_price = float(forecast_df["timesfm"].iloc[-1])
print(f"预测下一天的收盘价：${predicted_price:.2f}")
