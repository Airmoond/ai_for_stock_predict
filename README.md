# 基于 TimesFM 的 AI 股票预测模型

本项目是苯人的人工智能引论大作业，演示如何利用 Google 发布的 TimesFM 预训练模型，对指定股票进行最基础的“下一交易日收盘价”预测。项目目录下包含从数据获取到模型推断的完整流程示例。

---

## 文件说明

* **get\_data.py**
  负责调用 `yfinance` 库从 Yahoo Finance 下载指定股票（默认为 AAPL）的历史数据，并保存为 `aapl_stock.csv`。

* **prepare\_data.py**
  将 `aapl_stock.csv` 中的 “Close” 列提取出来，做最基础的序列切分与归一化处理，生成供模型训练或测试使用的 `stock_data.npz` 文件。

* **run\_model.py**
  核心推断脚本：直接读取 `aapl_stock.csv`，取最近最多 512 天的原始收盘价段作为上下文，通过 `timesfm` 包加载 “google/timesfm-1.0-200m” 预训练权重，调用 `forecast` 接口获得下一个时刻的收盘价预测并打印到终端。

* **aapl\_stock.csv**
  示例下载好的原始股票数据（CSV 格式），包含日期、开高低收、成交量等信息。

* **stock\_data.npz**
  `prepare_data.py` 处理后生成的 NumPy 打包文件，内部保存了训练集和测试集的序列矩阵与归一化所需的最大值。仅在需要对模型进行进一步微调时才会使用到它。

* **aapl\_us\_2025.csv**
  内附的 AAPL 数据示例，如果在使用 `yfinance` 下载时遇到限流或网络问题，可直接使用此默认数据来跑完整流程。

---

## 安装说明

### 环境要求

* **Python 3.8+**
* **conda 或 virtualenv**（强烈建议在独立环境中安装）
* 依赖包会列在 `requirements.txt` 中

### 安装步骤

1. 克隆或下载本项目到本地（但我现在其实还没上传GitHub所以...）：

   ```bash
   git clone <你的项目地址>
   cd ai-for-stock-predict
   ```

2. 创建并激活专属环境（以 conda 为例）：

   ```bash
   conda create -n timesfm python=3.10
   conda activate timesfm
   ```

3. 安装依赖包：

   ```bash
   pip install -r requirements.txt
   ```


4. 如果你只是想快速使用内置示例数据 `aapl_us_2025.csv`，也可以跳过 `yfinance`，但运行以下命令仍需安装 `timesfm`、`pandas`、`numpy`：

   ```bash
   pip install pandas numpy timesfm[torch]
   ```
  
---

## 使用方法

以下示例假设你已经在 `timesfm` 环境里，并且当前工作目录是项目根目录。

### 0. 极简版-速通本项目（懒人必看）

下面那些都是扯淡，其实我已经把数据和数据分析都做好了，只需要完成安装步骤之后:

运行

```bash
python run_model.py
```

就可以啦！

当然了安装依赖库有些可能会很慢，比如jax或者jaxlib，可以考虑用

```bash
conda install -c conda-forge jax jaxlib
```

### 1. 下载股票数据（可选）

如果网络正常且想获取最新数据，执行：

```bash
python get_data.py
```

此命令会从 Yahoo Finance 下载 AAPL 的历史数据并保存为 `aapl_stock.csv`。
**注意**：若遇限流或网络问题，可直接跳到第 3 步，使用仓库中已提供的 `aapl_us_2025.csv`，只需把脚本中读取文件名改为对应名称即可。

### 2. 数据预处理

把 “Close” 列提取并做最基础处理，生成训练/测试集文件：

```bash
python prepare_data.py
```

执行后会在项目目录下生成 `stock_data.npz`，其中包含：

* `X_train, X_test, y_train, y_test`：用于模型训练或微调的序列数据
* `max_close`：归一化时使用的最大收盘价

如果只想跑预测流程而不做微调，可以跳过这一步，直接进行第 3 步。

### 3. 运行预测脚本

```bash
python run_model.py
```

脚本逻辑简要说明：

1. 读取 `aapl_stock.csv`（或 `aapl_us_2025.csv`）中的 “Close” 列。
2. 取最近最多 512 天收盘价作为上下文（context）。
3. 通过 `timesfm.TimesFM.from_pretrained("google/timesfm-1.0-200m")` 加载预训练模型权重。
4. 调用 `forecast(inputs, freq=[0])` 获得“下一交易日”收盘价预测。
5. 在终端打印出预测结果，例如：

   ```
   预测下一天的收盘价：$172.45
   ```

---

## 项目结构

```
ai-for-stock-predict/
├── get_data.py          # 下载股票数据脚本（调用 yfinance）
├── prepare_data.py      # 数据预处理脚本：生成 stock_data.npz
├── run_model.py         # 模型推断脚本：加载 TimesFM 并预测下一天价格
├── aapl_stock.csv       # 从 Yahoo Finance 下载的原始股票数据
├── aapl_us_2025.csv     # 内置示例数据（备用，当 yfinance 下载失败时使用）
├── stock_data.npz       # prepare_data.py 处理后生成的 NumPy 打包文件
├── requirements.txt     # 项目依赖列表
└── README.md            # 本文档
```

---

## 常见问题

1. **无法导入 `yfinance`**

   * 确认已在 `timesfm` 环境下执行 `pip install yfinance`。
   * IDE（如 VSCode、PyCharm）中务必切换到相同环境的 Python 解释器。

2. **`ModuleNotFoundError: No module named 'timesfm'`**

   * 执行 `pip install timesfm[torch]`，并确保脚本运行时使用的解释器是安装了该包的环境。

3. **模型下载过慢 / 卡住**

   * TimesFM 预训练权重大约数百 MB，首次运行会自动从 Hugging Face 下载，需要耐心等待。
   * 若网络不佳，可考虑使用 VPN 或离线下载模型文件后手动放到缓存路径。

4. **只想使用示例数据，不下载最新数据**

   * 直接将脚本中的 `aapl_stock.csv` 改为 `aapl_us_2025.csv`，或将两者同名放置，跳过 `get_data.py` 步骤。

5. **为什么要用 `stock_data.npz`？**

   * 该文件仅在后续对 TimesFM 进行“微调训练”时使用。若当前只做一次零样本预测，可不关心此文件。

---

## 致谢

* 感谢 Google Research 发布的 [TimesFM](https://github.com/google-research/timesfm) 模型。
* 感谢 Hugging Face 平台提供的 `timesfm` Python 包和模型托管服务。
* 本项目仅作示例用途，不作为投资建议，请在实际操作中谨慎评估风险。
