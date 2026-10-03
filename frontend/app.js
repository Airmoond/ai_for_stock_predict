"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { ticker: "ICBC_A", view: "overview", range: 60, data: null, request: 0, job: null, timer: null, kind: "analysis" };
const icons = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  chart: '<path d="M3 3v18h18M7 14l4-5 4 3 6-7"/>',
  news: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h4v4H7zM15 8h2M15 12h2M7 16h10"/>',
  layers: '<path d="m12 3 10 6-10 6L2 9zM2 14l10 6 10-6M2 18l10 6 10-6"/>',
  spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5zM20 3v4M18 5h4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5"/>',
  price: '<path d="M3 17 9 11l4 3 8-10M15 4h6v6"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 10v4"/>',
  shield: '<path d="M12 3 3 7v6c0 5 9 9 9 9s9-4 9-9V7zM8 12l3 3 5-6"/>',
};
function icon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.chart}</svg>`; }
function esc(value) { return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function num(value, digits = 2) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—"; }
function signed(value, digits = 3) { return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "—"; }
function pct(value, digits = 2) { return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "—"; }
function tone(value) { return value == null ? "" : value < 0 ? "negative" : value > 0 ? "positive" : ""; }
function date(value) { return value ? String(value).slice(0, 10).replaceAll("-", ".") : "—"; }
function modelLabel(model) { return model === "naive" ? "收盘价基准 · 非 AI" : model === "timesfm" ? "TimesFM" : model || "未运行"; }
function metrics() { return state.data?.report?.metrics || {}; }
function weightLabels() {
  const weights = state.data?.report?.weights || { forecast: 0.55, sentiment: 0.25, technical: 0.20 };
  return `价格 ${num(weights.forecast * 100, 0)}% · 舆情 ${num(weights.sentiment * 100, 0)}% · 技术 ${num(weights.technical * 100, 0)}%`;
}
function currentModel() { return state.data?.report?.model?.name || (state.data?.legacy ? "timesfm（旧版）" : null); }
function lastPrice() { return state.data?.prices.at(-1); }
function dayChange() { const rows = state.data?.prices || []; return rows.length > 1 ? rows.at(-1).close / rows.at(-2).close - 1 : null; }
async function api(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败，请稍后重试");
  return result;
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $("#toast").hidden = true; }, 3500);
}
async function load(report = "") {
  const request = ++state.request;
  $("#content").setAttribute("aria-busy", "true");
  $("#download-report").disabled = true;
  try {
    const data = await api(`/api/dashboard?ticker=${state.ticker}${report ? `&report=${encodeURIComponent(report)}` : ""}`);
    if (request !== state.request) return;
    state.data = data;
    renderHeader();
    render();
    if (data.active_job && !state.job) monitor(data.active_job);
  } catch (error) {
    if (request !== state.request) return;
    state.data = null;
    $("#content").innerHTML = empty("暂时无法读取研究数据", error.message, "重新加载", "reload", "shield");
    $("#data-banner").classList.add("error");
    $("#data-banner").innerHTML = `${icon("shield")}<span>${esc(error.message)}</span>`;
  } finally {
    if (request === state.request) $("#content").removeAttribute("aria-busy");
  }
}
function renderHeader() {
  const data = state.data;
  $("#ticker-select").value = data.ticker;
  $("#stock-meta").textContent = `${data.stock.symbol} · ${data.stock.market} · ${data.stock.currency}`;
  $("#report-select").innerHTML = data.reports.length ? data.reports.map((r) => `<option value="${esc(r.id)}">${date(r.data_as_of)} · ${esc(modelLabel(r.model))}</option>`).join("") : '<option value="">暂无报告</option>';
  $("#report-select").value = data.selected_report || "";
  $("#report-select").disabled = !data.reports.length;
  $("#download-report").disabled = !data.report;
  $("#data-banner").classList.remove("error");
  const last = data.prices.at(-1);
  const message = last ? `数据截至 ${date(last.ds)} · ${modelLabel(currentModel())}${data.legacy ? " · 历史报告，预测不展示" : ""}` : "暂无本地数据，请联网运行分析。";
  $("#data-banner").innerHTML = `${icon("clock")}<span>${esc(message)}</span>`;
}
const views = {
  overview: ["研究总览", "", "工银察势", "从市场舆情到持仓解读的智能陪伴助手"],
  market: ["行情与预测", "", "行情与预测", "看见价格变化，读懂趋势信号。"],
  sentiment: ["舆情分析", "", "舆情分析", "跟踪市场情绪，理解新闻影响。"],
  backtest: ["回测评估", "", "回测评估", "以历史数据检验研究判断。"],
};
function navigate(view) {
  state.view = view;
  const labels = views[view];
  $("#breadcrumb-view").textContent = labels[0];
  $("#page-title").textContent = labels[2];
  $("#page-subtitle").textContent = labels[3];
  document.querySelectorAll("[data-view]").forEach((el) => { const active = el.dataset.view === view; el.classList.toggle("active", active); if (active) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current"); });
  render();
}
function metric(label, value, foot, type = "", symbol = "chart") {
  return `<article class="metric-card" title="${esc(foot.replace(/<[^>]+>/g, " "))}"><div class="metric-label">${label}</div><div class="metric-value ${type}">${value}</div>${symbol === "price" && foot.includes("较前一交易日") ? `<div class="metric-foot">${foot}</div>` : ""}</article>`;
}
function summaryCards() {
  const data = state.data, m = metrics(), last = lastPrice(), change = dayChange();
  const count = data.forecast.length;
  const returnValue = !data.legacy ? m.exp_return : null;
  return `<div class="metrics-grid">${metric("收盘价", `${num(last?.close)}<small>${data.stock.currency}</small>`, `<span class="${tone(change)}">${pct(change)}</span><span>较前一交易日</span>`, "", "price")}${metric(`${count || "未来"} 日预测变化`, pct(returnValue), esc(modelLabel(currentModel())), tone(returnValue), "target")}${metric("综合评分", signed(m.combo_score), esc(weightLabels()), tone(m.combo_score), "layers")}${metric("研究评级", esc(data.report?.recommendation?.rating || "待分析"), `数据日期 ${date(data.data_as_of)}`, "text-value", "shield")}</div>`;
}
function empty(title, detail, label = "", action = "analysis", symbol = "chart") {
  return `<div class="panel empty-state"><div class="empty-icon">${icon(symbol)}</div><h3>${esc(title)}</h3><p>${esc(detail)}</p>${label ? `<button class="button primary" data-action="${action}">${esc(label)} ↗</button>` : ""}</div>`;
}
function panelHead(title, subtitle = "", controls = "") {
  return `<div class="panel-head"><h3>${title}</h3>${controls}</div>`;
}
function average(period) {
  const prices = state.data.prices;
  let sum = 0;
  return prices.map((r, i) => { sum += r.close; if (i >= period) sum -= prices[i - period].close; return i >= period - 1 ? sum / period : null; });
}
let chartId = 0;
function chart(rows, series, label, options = {}) {
  if (rows.length < 2) return '<div class="empty-state"><p>至少需要两个有效数据点来绘制走势图。</p></div>';
  const mobile = window.matchMedia("(max-width: 760px)").matches;
  const width = mobile ? 360 : 700, height = mobile ? 230 : 270, left = mobile ? 40 : 46, right = 18, top = 22, bottom = 33;
  const values = rows.flatMap((r) => series.map((s) => r[s.key])).filter((n) => typeof n === "number" && Number.isFinite(n));
  if (!values.length) return '<div class="empty-state"><p>暂无可绘制的数据。</p></div>';
  const min = Math.min(...values), max = Math.max(...values), padding = Math.max((max - min) * 0.18, max * 0.003, 0.005);
  const lo = min - padding, hi = max + padding;
  const plotWidth = width - left - right;
  // Reserve enough room to read the short forecast even against a year of history.
  const split = options.forecastStart;
  const x = (i) => split > 0
    ? left + (i <= split ? i / split * plotWidth * 0.88 : plotWidth * (0.88 + 0.12 * (i - split) / (rows.length - 1 - split)))
    : left + i / (rows.length - 1) * plotWidth;
  const y = (v) => top + (hi - v) / (hi - lo) * (height - top - bottom);
  const id = `gradient-${++chartId}`;
  let svg = `<svg class="chart" role="img" aria-label="${esc(label)}" viewBox="0 0 ${width} ${height}"><title>${esc(label)}</title><defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#ad2333" stop-opacity=".14"/><stop offset="1" stop-color="#ad2333" stop-opacity="0"/></linearGradient></defs>`;
  for (let i = 0; i < 5; i++) {
    const value = lo + (hi - lo) * i / 4;
    svg += `<line class="chart-grid" x1="${left}" x2="${width - right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left - 9}" y="${y(value) + 3}" text-anchor="end">${value.toFixed(2)}</text>`;
  }
  if (options.forecastStart != null) {
    const start = x(options.forecastStart);
    svg += `<rect x="${start}" y="${top}" width="${width - right - start}" height="${height - top - bottom}" fill="#fcf0f0"/><line x1="${start}" x2="${start}" y1="${top}" y2="${height - bottom}" stroke="#dbafb5" stroke-dasharray="4 5"/><text x="${start + 8}" y="${top + 14}" style="fill:#ad2333">预测</text>`;
  }
  series.forEach((s, seriesIndex) => {
    let points = [], paths = [];
    rows.forEach((r, i) => {
      if (typeof r[s.key] === "number" && Number.isFinite(r[s.key])) points.push([x(i), y(r[s.key])]);
      else if (points.length) { paths.push(points); points = []; }
    });
    if (points.length) paths.push(points);
    paths.forEach((p) => {
      const line = p.map((point, i) => `${i ? "L" : "M"}${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(" ");
      if (seriesIndex === 0 && options.fill) svg += `<path d="${line} L${p.at(-1)[0]},${height - bottom} L${p[0][0]},${height - bottom} Z" fill="url(#${id})"/>`;
      svg += `<path d="${line}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round" ${s.dashed ? 'stroke-dasharray="5 5"' : ""}/>`;
    });
  });
  svg += `<line class="chart-axis" x1="${left}" x2="${width - right}" y1="${height - bottom}" y2="${height - bottom}"/>`;
  const ticks = mobile ? 3 : 5;
  for (let i = 0; i < ticks; i++) {
    const index = Math.round((rows.length - 1) * i / (ticks - 1));
    svg += `<text x="${x(index)}" y="${height - 12}" text-anchor="${i === 0 ? "start" : i === ticks - 1 ? "end" : "middle"}">${esc(date(rows[index].ds))}</text>`;
  }
  rows.forEach((r, i) => {
    const value = series.map((s) => r[s.key]).find((n) => typeof n === "number" && Number.isFinite(n));
    if (value == null) return;
    const title = `${date(r.ds)}  ${series.filter((s) => r[s.key] != null).map((s) => `${s.name} ${num(r[s.key], options.digits || 2)}`).join(" · ")}`;
    svg += `<circle cx="${x(i)}" cy="${y(value)}" r="7" fill="transparent" data-tooltip="${esc(title)}"><title>${esc(title)}</title></circle>`;
  });
  return `<div class="chart-wrap">${svg}</svg><div class="chart-tooltip" hidden></div></div>`;
}
function pricePanel(full = false) {
  const data = state.data;
  const offset = Math.max(0, data.prices.length - state.range), sma10 = average(10), sma30 = average(30);
  const history = data.prices.slice(offset).map((r, i) => ({ ds: r.ds, close: r.close, sma10: sma10[offset + i], sma30: sma30[offset + i], forecast: null }));
  const rows = [...history];
  if (data.forecast.length && rows.length) {
    rows.at(-1).forecast = rows.at(-1).close;
    data.forecast.forEach((r) => rows.push({ ds: r.ds, close: null, sma10: null, sma30: null, forecast: r.price }));
  }
  const controls = `<div class="segmented" aria-label="行情时间范围">${[60, 120, 250].map((n) => `<button data-range="${n}" class="${state.range === n ? "selected" : ""}" aria-pressed="${state.range === n}">${n === 60 ? "近 3 月" : n === 120 ? "近 6 月" : "近 1 年"}</button>`).join("")}</div>`;
  const series = [{ key: "close", name: "收盘", color: "#ad2333", width: 2.5 }, { key: "sma30", name: "MA30", color: "#cbb2a4", width: 1.5 }, { key: "forecast", name: "预测", color: "#cc7180", dashed: true }];
  if (full) series.splice(1, 0, { key: "sma10", name: "MA10", color: "#a59aa5", width: 1.2 });
  return `<section class="panel ${full ? "full-panel" : ""}">${panelHead("价格走势", "", controls)}<div class="chart-legend"><span><i class="legend-dot"></i>收盘价</span>${full ? '<span><i class="legend-dot" style="background:#a59aa5"></i>MA10</span>' : ""}<span><i class="legend-dot secondary-dot"></i>MA30</span><span><i class="legend-line"></i>${data.forecast.length ? "预测" : "暂无预测"}</span></div>${chart(rows, series, "工商银行历史收盘价、移动均线与预测走势", { fill: true, forecastStart: data.forecast.length ? history.length - 1 : null })}</section>`;
}
function signalPanel() {
  const m = metrics(), rating = state.data.report?.recommendation?.rating || "待分析";
  const angle = Number.isFinite(m.combo_score) ? Math.max(12, Math.min(360, (m.combo_score + 1) * 180)) : 0;
  const rows = [["价格预测", m.tfm_signal], ["新闻舆情", m.senti_score], ["技术指标", m.tech_score]];
  return `<section class="panel signal-panel">${panelHead("综合信号")}<div class="signal-score"><div class="score-circle" style="--score-angle:${angle}deg"><div class="score-inner"><strong>${signed(m.combo_score)}</strong><small>综合评分</small></div></div><div class="signal-verdict"><strong>${esc(rating)}</strong><p>评分范围 −1 至 +1<br>不代表上涨概率</p></div></div><div class="signal-bars">${rows.map(([label, value]) => { const v = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0; const width = Math.abs(v) * 50; return `<div class="signal-row"><span>${label}</span><div class="signal-track"><span class="${v < 0 ? "negative" : ""}" style="width:${width}%;left:${v < 0 ? 50 - width : 50}%"></span></div><b class="${tone(value)}">${signed(value, 2)}</b></div>`; }).join("")}</div></section>`;
}
function forecastPanel() {
  const rows = state.data.forecast, last = lastPrice()?.close;
  return `<section class="panel">${panelHead("未来交易日预测")}<div class="panel-body">${rows.length ? `<div class="forecast-list">${rows.map((r) => `<div class="forecast-row"><span>${date(r.ds)}</span><strong>${num(r.price)}</strong><span>${pct(last ? r.price / last - 1 : null)}</span></div>`).join("")}</div>` : '<div class="explanation">暂无有效预测，请重新运行分析。</div>'}</div></section>`;
}
function technicalPanel() {
  const t = state.data.report?.technical || {};
  const items = [["MA30", t.sma30, "30 日均线", 3], ["RSI 14", t.rsi14, "相对强弱指数", 1], ["MACD", t.macd_hist, "柱值", 4], ["成交量比", t.volume_ratio, "对比此前 20 日", 2]];
  return `<section class="panel">${panelHead("技术指标")}<div class="panel-body"><div class="technical-grid">${items.map(([label, value, desc, digits]) => `<div class="technical-item" title="${esc(desc)}"><span>${label}</span><strong>${num(value, digits)}</strong></div>`).join("")}</div></div></section>`;
}
function sentimentMini() {
  const rows = state.data.sentiment, value = metrics().senti_score;
  return `<section class="panel">${panelHead("近期新闻情绪", "近 30 个自然日内的聚合记录", '<button class="panel-link" data-go="sentiment">查看详情 ↗</button>')}<div class="panel-body"><div class="news-summary"><strong class="${tone(value)}">${signed(value, 2)}</strong><div><span class="badge ${value < 0 ? "red" : "muted"}">${value == null ? "暂无信号" : value < -0.1 ? "情绪偏弱" : value > 0.1 ? "情绪偏强" : "情绪中性"}</span><small>${rows.length ? `${rows.reduce((sum, r) => sum + r.volume, 0)} 篇新闻 · ${rows.length} 个记录日` : "近期暂无聚合新闻"}</small></div></div><div class="sentiment-strip" aria-label="近期舆情指数变化">${rows.slice(-22).map((r) => `<span class="${r.index < 0 ? "neg" : ""}" style="height:${Math.max(4, Math.abs(r.index) * 37)}px" title="${esc(r.ds)} · ${signed(r.index)}"></span>`).join("")}</div><div class="mini-caption"><span>${date(rows[0]?.ds)}</span><span>${date(rows.at(-1)?.ds)}</span></div></div></section>`;
}
function evidencePanel() {
  const rec = state.data.report?.recommendation;
  return `<section class="panel">${panelHead("研究依据")}<div class="panel-body"><p class="explanation">${esc(interpretation())}</p><details class="explanation-details"><summary>查看计算依据</summary><p class="explanation">${esc(rec?.note || "暂无计算结果。")}</p><p class="explanation">${esc(weightLabels())}</p></details><ul class="conditions">${(rec?.trigger_evaluations || []).map((r) => `<li><span>${esc(r.condition)}</span><span class="badge ${!r.evaluated ? "neutral" : r.triggered ? "red" : "muted"}">${!r.evaluated ? "无法评估" : r.triggered ? "已触发" : "未触发"}</span></li>`).join("")}</ul></div></section>`;
}
function interpretation() {
  const m = metrics(), rec = state.data.report?.recommendation;
  if (!rec) return "运行分析后，即可查看当前市场信号的综合解读。";
  if (state.data.legacy) return "当前展示历史报告。重新运行分析，可查看经过交易日校验的预测与研究依据。";
  const direction = (value) => value == null ? "暂无信号" : Math.abs(value) < 0.05 ? "平稳" : value > 0 ? "偏强" : "偏弱";
  return `当前评级为${rec.rating}。价格预测${direction(m.tfm_signal)}，${m.sentiment_available === false ? "近期舆情不足" : `新闻舆情${direction(m.senti_score)}`}，技术信号${direction(m.tech_score)}。`;
}
function interpretationPanel() {
  return `<section class="interpretation-panel"><div><h3>研究解读</h3><p>${esc(interpretation())}</p></div><button class="panel-link" data-go="market">查看依据 ↗</button></section>`;
}
function warningsPanel() {
  const data = state.data;
  return `<section class="panel">${panelHead("数据说明")}<div class="panel-body"><ul class="warning-list">${data.warnings.length ? data.warnings.map((w) => `<li>${esc(w)}</li>`).join("") : '<li>本报告未记录数据质量警告。</li>'}</ul><p class="source-info">${date(data.data_as_of)} · ${esc(modelLabel(currentModel()))}</p></div></section>`;
}
function marketView() {
  return `${summaryCards()}${pricePanel(true)}<div class="detail-grid"><div>${forecastPanel()}<div style="margin-top:20px">${evidencePanel()}</div></div><div>${technicalPanel()}<div style="margin-top:20px">${warningsPanel()}</div></div></div>`;
}
function sentimentView() {
  const rows = state.data.sentiment, m = metrics();
  const total = rows.reduce((sum, r) => sum + r.volume, 0);
  const average = total ? rows.reduce((sum, r) => sum + r.mean * r.volume, 0) / total : null;
  return `<div class="metrics-grid">${metric("当前舆情信号", signed(m.senti_score), m.sentiment_available ? "已纳入综合评分" : "暂无有效近期舆情", tone(m.senti_score), "news")}${metric("近期新闻数量", `${total || "—"}<small>篇</small>`, "行情日期前 30 个自然日", "", "news")}${metric("加权情绪均分", num(average, 3), "0 为负面 · 1 为正面", "", "target")}${metric("舆情更新日期", date(m.sentiment_as_of || rows.at(-1)?.ds), "以报告中的有效日期为准", "text-value", "clock")}</div>${rows.length ? `<section class="panel full-panel">${panelHead("新闻舆情指数走势", "有方向的情绪指数 · −1 至 +1")}${chart(rows, [{ key: "index", name: "舆情指数", color: "#ad2333" }], "最近30个自然日新闻舆情指数", { digits: 3 })}<div class="insight-note">情绪均分映射为有方向的指数，并按新闻量调整强度。超过 7 日的舆情不进入当前综合评分。</div></section><section class="panel">${panelHead("每日舆情记录", "现有缓存提供聚合数据，暂无逐条新闻标题")}<div class="table-wrap"><table><thead><tr><th>发布日期</th><th class="table-number">新闻数量</th><th class="table-number">情绪均分</th><th class="table-number">舆情指数</th><th>情绪方向</th></tr></thead><tbody>${[...rows].reverse().map((r) => `<tr><td>${date(r.ds)}</td><td class="table-number">${r.volume}</td><td class="table-number">${num(r.mean, 3)}</td><td class="table-number ${tone(r.index)}">${signed(r.index)}</td><td><span class="badge ${r.index < 0 ? "red" : "muted"}">${r.index < 0 ? "负面" : r.index > 0 ? "正面" : "中性"}</span></td></tr>`).join("")}</tbody></table></div></section>` : empty("近期暂无新闻聚合记录", "可切换到联网模式获取新闻，并生成新分析报告。", "运行分析", "analysis", "news")}<div class="detail-grid">${evidencePanel()}${warningsPanel()}</div>`;
}
function backtestView() {
  const test = state.data.backtest;
  const controls = '<button class="button secondary" data-action="backtest">运行新的回测 ↗</button>';
  if (!test) return empty("用一次回测，验证你的假设", "尚无此标的的回测记录。使用本地历史行情运行滚动回测，比较预测与策略表现。", "运行回测", "backtest", "layers");
  const s = test.summary;
  const rows = [["均价误差 MAE", s.forecast?.mae, s.naive_baseline?.mae, 4], ["均方根误差 RMSE", s.forecast?.rmse, s.naive_baseline?.rmse, 4], ["百分比误差 MAPE", s.forecast?.mape, s.naive_baseline?.mape, "percent"], ["方向准确率", s.forecast?.direction_accuracy, s.naive_baseline?.direction_accuracy, "percent"]];
  return `<div class="metrics-grid">${metric("策略累计净收益", pct(s.strategy?.total_return), "已扣除交易成本 · 未年化", tone(s.strategy?.total_return), "price")}${metric("买入持有净收益", pct(s.buy_and_hold?.total_return), "同区间 · 同样计入成本", tone(s.buy_and_hold?.total_return), "chart")}${metric("策略最大回撤", pct(s.strategy?.max_drawdown), "回测区间内的峰值损失", "negative", "shield")}${metric("滚动回测样本", `${s.samples}<small>个</small>`, `预测长度 ${s.horizon_sessions} 个交易观测`, "", "layers")}</div><section class="panel full-panel">${panelHead("策略与基准净值", `${date(s.first_signal_date)} — ${date(s.last_signal_date)} · ${esc(modelLabel(s.model))}`, controls)}<div class="chart-legend"><span><i class="legend-dot"></i>研究策略</span><span><i class="legend-dot secondary-dot"></i>买入持有</span></div>${chart(test.curve, [{ key: "strategy", name: "策略净值", color: "#ad2333" }, { key: "benchmark", name: "买入持有", color: "#a99a9f" }], "滚动回测策略与买入持有净值", { digits: 4 })}<div class="chart-foot"><span>每边成本 ${s.cost_bps_per_side} 基点</span><span>${date(s.first_signal_date)} — ${date(s.last_signal_date)}</span></div></section><div class="detail-grid"><section class="panel">${panelHead("预测误差比较", "与最后收盘价基准在相同样本中比较")}<div class="table-wrap"><table><thead><tr><th>指标</th><th class="table-number">${esc(modelLabel(s.model))}</th><th class="table-number">收盘价基准</th></tr></thead><tbody>${rows.map(([label, a, b, digits]) => `<tr><td>${label}</td><td class="table-number">${digits === "percent" ? (a == null ? "—" : `${num(a * 100)}%`) : num(a, digits)}</td><td class="table-number">${digits === "percent" ? (b == null ? "—" : `${num(b * 100)}%`) : num(b, digits)}</td></tr>`).join("")}</tbody></table></div><div class="insight-note">${s.model === "naive" ? "当前为非 AI 基准回测，预测结果与收盘价基准一致。" : "当前展示 TimesFM 历史样本评估。"}方向指标使用 ${s.forecast?.direction_samples ?? "—"} 个非持平样本。</div></section><section class="panel">${panelHead("回测方法与边界", "理解结果，也理解限制")}<div class="panel-body"><p class="explanation">信号在 T 日收盘后产生，T+1 收盘成交，计算随后一日收益。回测不包含舆情；历史 RSS 聚合缺少可验证的采集时点。</p><ul class="warning-list">${(s.method?.limits || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul><p class="source-info">持仓 ${s.strategy?.invested_sessions ?? "—"} 个观测 · 仓位变化 ${s.strategy?.turnover ?? "—"} 次</p></div></section></div>`;
}
function render() {
  if (!state.data) return;
  let html;
  if (!state.data.prices.length && state.view !== "backtest") html = empty("为这个标的建立第一份研究报告", "本地尚无有效行情。选择联网更新，获取数据并开始分析。", "运行分析", "analysis");
  else if (state.view === "market") html = marketView();
  else if (state.view === "sentiment") html = sentimentView();
  else if (state.view === "backtest") html = backtestView();
  else html = `${summaryCards()}<div class="dashboard-grid">${pricePanel()}${signalPanel()}</div>${interpretationPanel()}${state.data.warnings.length ? `<details class="overview-details"><summary>数据说明</summary>${warningsPanel()}</details>` : ""}`;
  $("#content").innerHTML = `<div class="view-entrance">${html}</div>`;
}
function openAnalysis(kind = "analysis") {
  state.kind = kind;
  $("#dialog-title").textContent = kind === "backtest" ? "开始回测" : "开始分析";
  $("#submit-analysis").textContent = kind === "backtest" ? "运行滚动回测 ↗" : "生成分析报告 ↗";
  $("#analysis-mode").disabled = kind === "backtest";
  if (kind === "backtest") $("#analysis-mode").value = "offline";
  if (!state.data?.prices.length && kind === "analysis") $("#analysis-mode").value = "live";
  $("#form-error").hidden = true;
  $("#analysis-dialog").showModal();
}
function renderJob(job) {
  const el = $("#job-status");
  el.hidden = false;
  el.classList.toggle("failed", job.status === "failed");
  el.innerHTML = job.status === "running" ? `<div class="spinner"></div><span>${esc(job.message)} ${job.model === "timesfm" ? "首次运行可能需要下载模型权重。" : "你可以继续查看已有报告。"}</span>` : job.status === "failed" ? `<details open><summary>任务未完成 · 点击查看原因</summary><pre>${esc(job.message)}</pre></details>` : `${icon("shield")}<span>${esc(job.message)} · 已更新本地研究数据</span>`;
  $("#open-analysis").disabled = job.status === "running";
}
function monitor(job) {
  state.job = job;
  renderJob(job);
  clearTimeout(state.timer);
  state.timer = setTimeout(async () => {
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(job.id)}`);
      renderJob(result);
      if (result.status === "running") monitor(result);
      else {
        state.job = null;
        if (result.status === "completed") {
          // Only reload the selected ticker; a user may have switched while waiting.
          await load();
          toast(result.message);
        }
      }
    } catch (error) {
      state.job = null;
      renderJob({ status: "failed", message: `无法获取任务状态：${error.message}。刷新页面可重新连接。` });
    }
  }, 1400);
}
$("#analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.data) return;
  $("#submit-analysis").disabled = true;
  $("#form-error").hidden = true;
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json", "X-Research-Token": state.data.token }, body: JSON.stringify({ ticker: state.ticker, model: $("#analysis-model").value, mode: $("#analysis-mode").value, horizon: Number($("#analysis-horizon").value), kind: state.kind }) });
    $("#analysis-dialog").close();
    monitor(job);
  } catch (error) {
    $("#form-error").textContent = error.message;
    $("#form-error").hidden = false;
  } finally { $("#submit-analysis").disabled = false; }
});
$("#analysis-model").addEventListener("change", () => { $("#model-hint").textContent = $("#analysis-model").value === "timesfm" ? "使用 AI 模型预测；需已配置 TimesFM 运行环境。" : "以最后收盘价作为未来价格的比较基准。"; });
$("#open-analysis").addEventListener("click", () => openAnalysis());
$("#close-dialog").addEventListener("click", () => $("#analysis-dialog").close());
$("#cancel-dialog").addEventListener("click", () => $("#analysis-dialog").close());
$("#report-select").addEventListener("change", (event) => load(event.target.value));
function switchTicker(ticker) {
  if (ticker === state.ticker) return;
  state.ticker = ticker;
  document.querySelectorAll("[data-ticker]").forEach((el) => el.classList.toggle("selected", el.dataset.ticker === ticker));
  load();
}
$("#ticker-select").addEventListener("change", (event) => switchTicker(event.target.value));
$("#download-report").addEventListener("click", () => {
  if (!state.data?.report) return;
  const blob = new Blob([JSON.stringify(state.data.report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = `${state.ticker}_${state.data.data_as_of}_report.json`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("当前分析报告已导出");
});
document.addEventListener("click", (event) => {
  const view = event.target.closest("[data-view]");
  if (view) navigate(view.dataset.view);
  const go = event.target.closest("[data-go]");
  if (go) { navigate(go.dataset.go); $("#main").scrollIntoView({ behavior: "smooth" }); }
  const range = event.target.closest("[data-range]");
  if (range) { state.range = Number(range.dataset.range); render(); }
  const ticker = event.target.closest("[data-ticker]");
  if (ticker) switchTicker(ticker.dataset.ticker);
  const action = event.target.closest("[data-action]");
  if (action) { if (action.dataset.action === "reload") load(); else openAnalysis(action.dataset.action === "backtest" ? "backtest" : "analysis"); }
});
document.addEventListener("pointerover", (event) => {
  const point = event.target.closest("[data-tooltip]");
  if (!point) return;
  const wrap = point.closest(".chart-wrap"), tip = wrap.querySelector(".chart-tooltip");
  tip.textContent = point.dataset.tooltip; tip.hidden = false;
  const rect = wrap.getBoundingClientRect();
  tip.style.left = `${Math.max(8, Math.min(event.clientX - rect.left + 10, rect.width - tip.offsetWidth - 8))}px`;
  tip.style.top = `${Math.max(0, event.clientY - rect.top - 35)}px`;
});
document.addEventListener("pointerout", (event) => {
  if (event.target.closest("[data-tooltip]")) event.target.closest(".chart-wrap").querySelector(".chart-tooltip").hidden = true;
});
document.querySelectorAll("[data-icon]").forEach((el) => { el.innerHTML = icon(el.dataset.icon); });
window.matchMedia("(max-width: 760px)").addEventListener("change", () => render());
$("#today").textContent = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date()).replaceAll("/", ".");
navigate("overview");
load();
