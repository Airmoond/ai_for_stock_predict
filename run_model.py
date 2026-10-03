import pandas as pd
import timesfm
import torch
import matplotlib
import matplotlib.pyplot as plt

# ————【字体设置：解决中文和负号问题】————
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   
matplotlib.rcParams['axes.unicode_minus'] = False     

# ---------------------------------------
# 1. 读取 CSV 并预处理
# ---------------------------------------
df = pd.read_csv("aapl_stock.csv", parse_dates=["Date"])
df2 = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
df2["unique_id"] = "AAPL"

# 限制最近 512 条数据
if len(df2) > 512:
    df2 = df2.iloc[-512:].copy()

# ---------------------------------------
# 2. 加载模型并预测
# ---------------------------------------
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="torch",
        per_core_batch_size=32,
        horizon_len=10,      
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
# forecast_on_df 会返回一个 DataFrame，包含从“下一天”起连续 10 行的 ds（日期）和 timesfm（预测值）以及分位数列等
forecast_df = tfm.forecast_on_df(
    inputs=df2,
    freq="D",          # 数据是按天频率
    value_name="y",    # 原始收盘价列名
    num_jobs=1
)

# 打印方便检查
print("未来 10 天的预测结果：")
print(forecast_df[["ds", "timesfm"]])

# ---------------------------------------
# 3. 可视化：绘制历史收盘价 + 未来 10 天预测折线
# ---------------------------------------

# 3.1 取出历史数据
history_dates = df2["ds"]
history_prices = df2["y"]

# 3.2 取出未来 10 天的预测日期和预测值
pred_dates = forecast_df["ds"]            # 未来 10 个日期
pred_prices = forecast_df["timesfm"]      # 对应的预测收盘价

# 3.3 开始绘图
plt.figure(figsize=(12, 6))

# 画出历史收盘价折线
plt.plot(history_dates, history_prices,
         label="历史收盘价",
         color="tab:blue",
         linewidth=2)

# 画出未来 10 天的预测折线
plt.plot(pred_dates, pred_prices,
         label="未来10天预测收盘价",
         color="tab:orange",
         linewidth=2)

# 3.4 美化图表
plt.title("AAPL 收盘价 — 历史 vs. 未来10天预测", fontsize=18)
plt.xlabel("日期", fontsize=14)
plt.ylabel("收盘价 ($)", fontsize=14)

# 调整 x 轴的日期显示
plt.xticks(rotation=30)

plt.legend(loc="upper left", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

