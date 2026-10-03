# 工银察势——从市场舆情到持仓解读的智能陪伴助手

结合 **TimesFM 价格预测、新闻舆情与技术指标**，生成工商银行股票的短期分析报告，并通过滚动回测比较预测误差和策略表现。

项目提供中文股票研究看板和 Python 命令行脚本，适用于金融时间序列研究、课程实验和策略原型验证。默认分析工商银行 A 股，支持切换至港股。

| 市场 | 项目别名 | 行情代码 |
| --- | --- | --- |
| 工商银行 A 股 | `ICBC_A` | `601398.SS` |
| 工商银行港股 | `ICBC_H` | `1398.HK` |

## 目录

- [主要功能](#主要功能)
- [快速开始](#快速开始)
- [股票研究看板](#股票研究看板)
- [使用 TimesFM](#使用-timesfm)
- [项目结构](#项目结构)
- [运行分析](#运行分析)
- [分析方法](#分析方法)
- [报告与数据](#报告与数据)
- [滚动回测](#滚动回测)
- [测试与验证](#测试与验证)
- [常见问题](#常见问题)
- [当前限制](#当前限制)

## 主要功能

- **行情获取与缓存**：A 股使用 AKShare，港股使用 yfinance；检查日期范围与数据完整性，自动刷新过期缓存。
- **交易日预测**：根据交易所日历跳过周末和节假日，默认预测未来 3 个交易日。
- **中英文舆情**：从 Google News RSS 获取标题及摘要，中文使用 SnowNLP，英文使用 VADER。
- **技术分析**：计算 10/30 日均线、14 日 RSI 和 MACD。
- **信号融合**：生成综合评分、评级说明与触发条件检查结果。
- **数据质量标记**：显示行情截至日期、缓存来源、缺失交易日和获取错误。
- **滚动回测**：比较模型与价格基准，统计预测误差、含成本收益和最大回撤。
- **结构化输出**：行情与舆情保存为 CSV，分析报告和回测汇总保存为 JSON。

## 快速开始

以下 PowerShell 命令均在项目根目录执行。

### 1. 安装轻量环境

轻量分析及 `naive` 基准模式支持 Python 3.11 / 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 使用现有数据运行离线示例

仓库包含截至 **2025-10-10** 的 A 股历史行情。执行以下命令后，运行阶段不访问行情或新闻服务，也不下载模型权重：

```powershell
.\.venv\Scripts\python.exe analyze_ticker.py --offline --model naive --end 2025-10-10 --cache-dir reports --output-dir reports/validation
```

生成的报告位于：

```text
reports/validation/ICBC_A_naive_report.json
```

`naive` 表示“未来价格等于最后收盘价”的基准。报告会明确注明“非 AI”；它用于验证流程和比较预测效果。

### 3. 获取最新数据

```powershell
.\.venv\Scripts\python.exe analyze_ticker.py --model naive --refresh
```

此命令获取最新行情和新闻，将缓存及基准报告写入 `reports/`。默认模式为 TimesFM，因此在轻量环境中运行时需要显式指定 `--model naive`。

## 股票研究看板

在项目根目录启动：

```powershell
.\.venv\Scripts\python.exe webapp.py --open
```

浏览器访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。如端口被占用，可使用 `--port 8765`。按 `Ctrl+C` 停止服务。

看板本身只使用 Python 标准库和原生 HTML/CSS/JavaScript，无需安装 Node.js 或构建前端。首次打开直接读取 `reports/` 中的现有数据，不会自动联网或下载模型。

- **研究总览**：最后收盘价、预测均价变化、综合评分、研究评级和三类分项信号。
- **行情与预测**：3 月 / 6 月 / 1 年走势图、10/30 日均线、未来交易日预测与条件检查；悬停可读数值。
- **舆情分析**：近期新闻量、情绪走势和每日聚合记录，显示有效舆情日期。
- **回测评估**：策略与买入持有的净值、预测误差、交易成本、回撤和样本区间。
- **分析操作**：切换 A/H 股、选择历史报告、导出 JSON，运行离线/联网分析或本地回测。

点击“运行分析”选择模型、数据来源和预测交易日数。默认使用明确标注“非 AI”的收盘价基准；离线分析的截止日期自动设为可用缓存的最后一个交易日。港股没有随仓库提供数据，需先联网运行分析。

后台任务使用启动看板的同一个 Python 环境。要使用 TimesFM，需改用 `.venv-timesfm\Scripts\python.exe webapp.py --open`。模型依赖缺失或行情获取失败时，页面显示错误原因。回测默认评估最近 120 个样本、每边成本 10 基点，沿用现有回测规则。

每次运行在 `reports/web/runs/` 中保存独立报告和配套缓存，不覆盖原始报告。页面优先加载生成时间最新的报告，也可以选择历史版本；图表按该报告的数据日期截断，避免混入后续行情。原有旧版 TimesFM 报告仍可查看，但其未经交易日校验的预测不会绘制，需重新分析生成新版结果。

服务仅监听本机地址，适合个人研究与项目演示；当前没有多人账户系统。图表中的价格预测和综合分均以研究结果展示，不代表已校准的上涨概率。

## 使用 TimesFM

项目保留原有预训练模型：

- 模型权重：`google/timesfm-1.0-200m-pytorch`
- Python 包：`timesfm[torch]==1.3.0`
- PyTorch 模式环境：**Python 3.11**
- 最大输入上下文：512 条交易观测
- 默认设备：CPU

为 TimesFM 创建独立环境：

```powershell
py -3.11 -m venv .venv-timesfm
.\.venv-timesfm\Scripts\python.exe -m pip install -r requirements-timesfm.txt
.\.venv-timesfm\Scripts\python.exe analyze_ticker.py --refresh
```

首次执行会从 Hugging Face 下载模型权重，需要网络连接和足够内存。具备相应 PyTorch CUDA 环境时，可添加 `--backend gpu`。

版本及环境要求参见 [Google 旧版模型说明](https://github.com/google-research/timesfm/blob/master/v1/README.md) 和 [官方依赖配置](https://github.com/google-research/timesfm/blob/master/v1/pyproject.toml)。

TimesFM 加载失败时会明确报错，不会自动切换为基准模型。当前验证覆盖接口适配；真实模型权重推理尚未在本次开发环境中验证。

## 项目结构

```text
.
├── analyze_ticker.py           # 命令行分析入口
├── webapp.py                   # 本地看板服务与后台分析任务
├── frontend/                   # 中文研究界面、图表和交互
├── config.py                  # 股票映射、路径、权重与阈值
├── fetch_price.py             # 行情获取、清洗和缓存
├── fetch_sentiment.py         # RSS 获取、中英文情绪评分
├── trading_calendar.py        # 交易所日历与已收盘交易日
├── indicators.py              # 均线、RSI、MACD 和技术条件
├── timesfm_model.py            # TimesFM 推理及 naive 基准
├── ensemble.py                # 信号融合与舆情异常检查
├── recommend.py               # 评级、依据与条件评估
├── backtest.py                # 滚动回测入口
├── storage.py                 # CSV / JSON 原子写入
├── requirements.txt           # 轻量运行依赖
├── requirements-timesfm.txt   # TimesFM 额外依赖
├── requirements-dev.txt       # 测试依赖
├── tests/
│   └── test_pipeline.py        # 回归与完整流程测试
└── reports/
    ├── ICBC_A_price.csv        # 原始历史行情缓存
    ├── ICBC_A_sentiment.csv    # 原始历史舆情聚合缓存
    └── ICBC_A_report.json      # 2025-10-10 的历史 TimesFM 报告
```

运行后还可生成 `reports/validation/` 和 `reports/backtests/`。它们及虚拟环境目录已加入 `.gitignore`，不要求随仓库分发。

## 运行分析

分析港股：

```powershell
.\.venv-timesfm\Scripts\python.exe analyze_ticker.py --ticker ICBC_H --refresh --strict-data
```

指定日期范围及预测长度：

```powershell
.\.venv-timesfm\Scripts\python.exe analyze_ticker.py --ticker ICBC_A --start 2023-01-01 --end 2025-10-10 --horizon 5 --offline
```

查看全部选项：

```powershell
.\.venv\Scripts\python.exe analyze_ticker.py --help
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--ticker` | `ICBC_A` | 选择 `ICBC_A` 或 `ICBC_H` |
| `--start` | `2020-01-01` | 行情起始日期 |
| `--end` | 最近已收盘交易日 | 包含结束日，排除尚未完成的当日行情 |
| `--horizon` | `3` | 预测交易日数量 |
| `--model` | `timesfm` | 选择 `timesfm` 或 `naive` |
| `--backend` | `cpu` | TimesFM 设备，支持 `cpu` / `gpu` |
| `--offline` | 关闭 | 只使用本地缓存 |
| `--refresh` | 关闭 | 强制刷新行情；与离线模式互斥 |
| `--strict-data` | 关闭 | 拒绝过期或存在缺失交易日的行情 |
| `--output-dir` | 项目中的 `reports/` | 报告输出目录 |
| `--cache-dir` | 与输出目录相同 | 行情与舆情缓存目录 |

默认路径以项目位置为基准；命令行指定的相对路径以当前运行目录为基准。更换输出目录后，如需读取原有缓存，应同时指定 `--cache-dir reports`。

## 分析方法

### 处理流程

```text
历史行情 ──→ TimesFM / naive 预测 ──┐
新闻 RSS ──→ 中英文舆情评分 ────────┼─→ 加权融合 ─→ 条件检查 ─→ JSON 报告
历史行情 ──→ 均线、RSI、MACD ───────┘
```

### 行情、缓存与交易日

A 股使用前复权日线，港股使用自动复权日线。读取时统一日期与字段，剔除非法价格、排序并去重，兼容 yfinance 多层列名。

每次加载均按请求区间筛选。缓存超过 24 小时、缺少最新已收盘交易日或缺少区间记录时，会尝试刷新。复权事件可能修订历史价格，因此刷新替换整个请求区间，避免直接拼接不同复权基准的数据。

联网失败时可回退至可用缓存，并在报告中记录原因。行情过期或缺失交易日时，评级显示“数据不足”；严格模式直接拒绝继续分析。少于 30 条有效行情时无法完成预测。

A 股使用 XSHG 日历，港股使用 XHKG 日历。收盘后等待 15 分钟才将当天作为完整日线。日历未覆盖所需未来日期时明确报错，需要更新日历依赖。

### 舆情评分

RSS 按标准 XML 解析，使用标题及可用摘要，过滤重复新闻、无效发布日期和行情截至日期之后的新闻。仅选取近 30 日数据；最近有效舆情超过 7 日时，其信号按 0 处理。

情绪均分从 `[0, 1]` 映射为有方向的指数，新闻量放大其方向：

```text
direction = 2 × sentiment_mean - 1
amplitude = 0.7 + 0.3 × min(log(1 + volume) / log(301), 1)
sentiment_index = direction × amplitude
```

负面新闻增多不会变成正面信号。旧版舆情 CSV 在读取时会依据原始均分和新闻量重新计算指数。获取失败或没有有效近期新闻时保留已有缓存。

### 技术指标与综合评分

技术分由三部分组成：

| 指标 | 条件 | 分值 |
| --- | --- | --- |
| 10/30 日均线 | 短均线高于 / 低于长均线 | +0.4 / −0.4 |
| RSI（14 日） | RSI < 30 / RSI > 70 | +0.3 / −0.3 |
| MACD 柱值 | 大于 / 小于 0 | +0.3 / −0.3 |

均线或 MACD 持平时对应分量为 0，前 30 条观测不产生技术评分。

```text
forecast_signal = clip(预测均价相对最后收盘价的变化 / 5%, -1, 1)
combo_score = 0.55 × forecast_signal
            + 0.25 × sentiment_index
            + 0.20 × technical_score
```

缺失舆情按 0 计分，不重新分配权重。权重与阈值集中在 [config.py](../config.py) 中配置。

| 综合分 | 基础评级 |
| --- | --- |
| ≥ 0.30 | 买入 |
| ≥ 0.10 且 < 0.30 | 谨慎增持/持有 |
| ≤ −0.30 | 回避/减仓 |
| 其他 | 观望 |

每次分析还检查以下条件：

- 连续日期的舆情变化较此前变化均值低超过 2σ：下调一级评级。需要至少 5 个历史变化样本及有效方差。
- 收盘价低于 30 日均线且 RSI < 30：转为回避/减仓。
- 成交量至少达到此前 20 日均量的 1.5 倍且 MACD 金叉：标记增强信号，仍按综合分评级。

数据质量检查优先于买卖评级。条件所需数据缺失时，报告注明“无法评估”。这些检查在脚本执行时完成。

## 报告与数据

| 产物 | 文件名示例 |
| --- | --- |
| 行情缓存 | `ICBC_A_price.csv` |
| 舆情缓存 | `ICBC_A_sentiment.csv` |
| TimesFM 分析报告 | `ICBC_A_report.json` |
| naive 分析报告 | `ICBC_A_naive_report.json` |
| 回测明细 | `ICBC_A_price_naive_backtest.csv` |
| 回测汇总 | `ICBC_A_price_naive_backtest.json` |

分析报告使用 `schema_version=2`，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `as_of` | 报告生成时间，带中国时区 |
| `data_as_of` | 最后一条实际行情日期 |
| `model` | 模型名称、权重来源、预测长度及输入长度 |
| `data_quality` / `warnings` | 缓存来源、离线状态、过期与缺失记录 |
| `weights` | 三类信号的融合权重 |
| `metrics` / `technical` | 综合评分、分项信号和技术指标 |
| `recommendation` | 评级、依据和条件评估结果 |
| `forecast` | 后续交易日期与估计价格 |

新版 `forecast` 替代旧版 `peek_forecast`。接入报告的其他程序需要按新版结构读取。

同一股票、同一模型再次运行会更新对应报告；基准模型使用独立文件名。需要保留不同日期的结果时，应指定不同的 `--output-dir`。CSV 与 JSON 使用原子写入，防止中途退出产生不完整文件。

仓库原有 `ICBC_A_report.json` 是 **2025-10-10 的历史 TimesFM 结果**。应通过 `data_as_of` 与模型信息区分新生成的分析和历史产物。

## 滚动回测

基准模式：

```powershell
.\.venv\Scripts\python.exe backtest.py --model naive --test-sessions 120 --cost-bps 10
```

TimesFM 模式：

```powershell
.\.venv-timesfm\Scripts\python.exe backtest.py --model timesfm --test-sessions 120 --cost-bps 10
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--price-csv` | `reports/ICBC_A_price.csv` | 本地行情文件，至少包含 `ds` 和 `Close` |
| `--model` | `timesfm` | 模型选择 |
| `--backend` | `cpu` | TimesFM 设备 |
| `--horizon` | `3` | 预测未来观测数量 |
| `--min-train` | `128` | 首次预测所需最少历史记录，最低 30 |
| `--test-sessions` | `120` | 最多评估最近多少个样本 |
| `--cost-bps` | `10` | 每边交易成本，10 基点为 0.1% |
| `--start` / `--end` | `2020-01-01` / 不限制结束日 | 行情筛选区间 |
| `--output-dir` | `reports/backtests/` | 回测输出目录 |

每个样本只使用截至信号日的历史记录，模型输入最多保留最近 512 条价格。预测目标是未来 `horizon` 条观测的平均价格。回测依赖 CSV 中的实际日期与记录，不重新获取行情。

策略规则为：买入或谨慎增持时持有，其他评级空仓。信号在 `t` 日收盘后产生，于 `t+1` 收盘成交，再计算 `t+1 → t+2` 的收益。每次仓位变化及最终平仓扣除成本，买入持有基准也扣除入场与退出成本。

输出包含：

- 预测 MAE、RMSE、MAPE，以及最后收盘价基准的相同指标。
- 方向准确率及有效方向样本数；实际目标价格持平的样本不参与该统计。
- 策略累计净收益、最大回撤、仓位变化次数与持仓观测数。
- 买入持有净收益与最大回撤。
- 每个样本的信号日期、预测目标、成交日期、持仓和收益明细。

**现有回测不包含舆情信号。** RSS 聚合缓存缺少可验证的历史采集时间，回测中按 0 处理并保留原权重。回测完整舆情策略需要先建立包含发布时间与采集时间的原始新闻档案。

## 测试与验证

安装测试依赖并执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests
```

本次改进验证记录：

| 验证项 | 结果 |
| --- | --- |
| 自动化测试 | 58 项通过（含看板数据与接口测试） |
| 离线完整分析 | 使用历史缓存生成独立 naive 报告 |
| 联网 A 股与 RSS 分析 | 行情截至 2026-09-30，请求区间无缺失交易日 |
| 未来交易日期 | 2026-10-08、2026-10-09、2026-10-12 |
| TimesFM 接口 | 已通过模拟模型适配测试；真实权重推理待验证 |

联网验证产物保存在本地 `reports/validation/live/`，该目录被 Git 忽略，其他环境需要重新运行才能生成。

使用截至 2025-10-10 的历史行情，对最近 120 个样本执行 naive 回测：

| 指标 | 结果 |
| --- | --- |
| 均价预测 MAE | 约 0.0748 |
| 策略累计净收益 | 约 0.34% |
| 买入持有累计净收益 | 约 10.37% |
| 每边交易成本 | 10 基点 |

这些数值来自基准模式，不是 TimesFM 效果评估。

## 常见问题

### 提示 TimesFM 需要 Python 3.11

检查运行命令是否使用 `.venv-timesfm` 中的 Python，并按 `requirements-timesfm.txt` 安装固定版本。轻量模式可使用 Python 3.12，但需要添加 `--model naive`。

### 离线模式提示没有可用缓存

检查 `--cache-dir`、股票别名和日期范围。仓库只包含 A 股历史缓存，港股需要先联网获取。更换输出目录时，缓存目录默认也会随之改变。

### 报告显示“数据不足”

查看 `data_as_of`、`data_quality` 和 `warnings`。行情可能过期或存在缺失交易日。可尝试联网刷新；查看历史结果时应明确指定 `--end`，不要将历史缓存当成最新数据。

### 交易日历提示日期超出覆盖范围

更新依赖后重新执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "exchange-calendars>=4.12,<5"
```

如果新版本仍未提供所需年份的节假日数据，则无法可靠生成该年份的预测日期。

### 新闻源不可用或没有近期舆情

脚本保留已有缓存并记录原因。没有 7 日内有效舆情时，该分量为 0；其余分析可以继续。网络恢复后重新执行联网分析即可重试。

## 当前限制

- TimesFM 真实模型推理和完整模型回测仍需在 Python 3.11 环境中验证。
- 新闻评分使用通用情绪工具，尚未进行金融领域专门训练或校准。
- 预测价格和综合分不代表已校准的上涨概率。
- 固定权重与阈值尚未经过系统参数优化。
- 复权历史价格可能事后修订；回测尚未模拟涨跌停、停牌成交约束和冲击成本。
- 当前提供本地研究看板、命令行分析及研究回测，尚未实现持续监控或自动交易。

