"""Local research dashboard. The web server itself needs only Python's stdlib."""
import argparse
import csv
import json
import math
import secrets
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FRONTEND = ROOT / "frontend"
CHINA = timezone(timedelta(hours=8))
TICKERS = {"ICBC_A": {"name": "工商银行", "symbol": "601398", "market": "上交所", "currency": "CNY"},
           "ICBC_H": {"name": "工商银行", "symbol": "01398", "market": "港交所", "currency": "HKD"}}


def load_json(path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_rows(path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return []


def prices(path, as_of=None):
    rows = {}
    for row in read_rows(path):
        day, close = row.get("ds", "")[:10], number(row.get("Close"))
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        if close is None or close <= 0 or (as_of and day > as_of):
            continue
        rows[day] = {"ds": day, "close": close,
                     "volume": number(row.get("volume", row.get("Volume"))),
                     "open": number(row.get("open", row.get("Open"))),
                     "high": number(row.get("high", row.get("High"))),
                     "low": number(row.get("low", row.get("Low")))}
    return [rows[day] for day in sorted(rows)]


def sentiment(path, as_of):
    if not as_of:
        return []
    start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    rows = {}
    for row in read_rows(path):
        day = row.get("ds", "")[:10]
        mean, volume = number(row.get("sentiment_mean")), number(row.get("volume"))
        if not start <= day <= as_of or mean is None or volume is None or not 0 <= mean <= 1 or volume < 0:
            continue
        # Recompute legacy positive-only indices using the pipeline's signed formula.
        index = (2 * mean - 1) * (0.7 + 0.3 * min(math.log1p(volume) / math.log(301), 1))
        rows[day] = {"ds": day, "mean": mean, "volume": volume, "index": index}
    return [rows[day] for day in sorted(rows)]


class ResearchStore:
    def __init__(self, report_dir=REPORTS):
        self.report_dir = Path(report_dir).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.jobs = {}
        self.active_job = None

    def files(self, pattern):
        for path in self.report_dir.rglob(pattern):
            parts = path.relative_to(self.report_dir).parts
            # Only completed research artifacts under web/runs belong to the UI.
            # Browser profiles, test fixtures and logs may share the web directory.
            if parts[0] == "web" and (len(parts) < 3 or parts[1] != "runs"):
                continue
            if path.resolve().is_relative_to(self.report_dir):
                yield path

    def artifacts(self, ticker, suffix="report"):
        found = []
        for path in self.files(f"{ticker}*_{suffix}.json"):
            data = load_json(path)
            if data and (suffix != "report" or data.get("ticker") == ticker):
                found.append((path, data))
        found.sort(key=lambda item: (str(item[1].get("as_of", item[1].get("last_signal_date", ""))),
                                     item[0].stat().st_mtime), reverse=True)
        return found

    def choose(self, ticker, report_id=None):
        reports = self.artifacts(ticker)
        if report_id:
            reports = [item for item in reports if item[0].relative_to(self.report_dir).as_posix() == report_id]
            if not reports:
                raise ValueError("报告不存在，请刷新报告列表")
        return reports[0] if reports else (None, None)

    def cache(self, ticker, report_path=None, kind="price"):
        name = f"{ticker}_{kind}.csv"
        if report_path and (report_path.parent / name).is_file():
            return report_path.parent / name
        candidates = list(self.files(name))
        if not candidates:
            return self.report_dir / name
        if report_path:
            # Historical report folders often share the original cache in reports/.
            for folder in report_path.parents:
                if folder == self.report_dir.parent:
                    break
                if (folder / name).is_file():
                    return folder / name
        if kind == "price":
            return max(candidates, key=lambda p: (prices(p)[-1]["ds"] if prices(p) else "", p.stat().st_mtime))
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def dashboard(self, ticker, report_id=None):
        if ticker not in TICKERS:
            raise ValueError("不支持的股票代码")
        report_path, report = self.choose(ticker, report_id)
        as_of = (report.get("data_as_of") or str(report.get("as_of", ""))[:10]) if report else None
        price_path = self.cache(ticker, report_path)
        history = prices(price_path, as_of)
        as_of = as_of or (history[-1]["ds"] if history else None)
        warnings = list(report.get("warnings", [])) if report else ["尚无分析报告，可先运行分析。"]
        legacy = bool(report and report.get("schema_version") != 2)
        if legacy:
            warnings.append("旧版历史报告缺少交易日校验，预测不展示；请重新运行分析生成新版报告。")
        if as_of and (not history or history[-1]["ds"] != as_of):
            warnings.append("行情缓存与报告日期不一致，图表仅展示现有缓存。")
        source = price_path.relative_to(self.report_dir).as_posix()
        senti_path = self.cache(ticker, report_path, "sentiment")
        tests = self.artifacts(ticker, "backtest")
        backtest = None
        if tests:
            test_path, summary = tests[0]
            detail = read_rows(test_path.with_suffix(".csv"))
            wealth, benchmark = 1.0, 1.0
            curve = []
            cost = (number(summary.get("cost_bps_per_side")) or 0) / 10000
            for i, row in enumerate(detail):
                strategy_return, market_return = number(row.get("strategy_return")), number(row.get("market_return"))
                if strategy_return is None or market_return is None:
                    continue
                wealth *= 1 + strategy_return
                benchmark *= 1 + market_return
                if i == 0:
                    benchmark *= 1 - cost
                if i == len(detail) - 1:
                    benchmark *= 1 - cost
                curve.append({"ds": row.get("return_date"), "strategy": wealth, "benchmark": benchmark})
            backtest = {"summary": summary, "curve": curve, "source": test_path.relative_to(self.report_dir).as_posix()}
        with self.lock:
            active = dict(self.jobs[self.active_job]) if self.active_job else None
        return {"ticker": ticker, "stock": TICKERS[ticker], "data_as_of": as_of,
                "prices": history[-520:], "sentiment": sentiment(senti_path, as_of),
                "report": report, "legacy": legacy, "forecast": [] if legacy else (report or {}).get("forecast", []),
                "warnings": warnings, "price_source": source, "backtest": backtest,
                "reports": [{"id": p.relative_to(self.report_dir).as_posix(), "as_of": r.get("as_of"),
                             "data_as_of": r.get("data_as_of", str(r.get("as_of", ""))[:10]),
                             "model": r.get("model", {}).get("name", "timesfm（旧版）")} for p, r in self.artifacts(ticker)],
                "selected_report": report_path.relative_to(self.report_dir).as_posix() if report_path else None,
                "token": self.token, "active_job": active}

    def start(self, payload):
        ticker, model, mode = payload.get("ticker"), payload.get("model"), payload.get("mode")
        horizon = payload.get("horizon", 3)
        if ticker not in TICKERS or model not in ("naive", "timesfm") or mode not in ("offline", "live"):
            raise ValueError("股票、模型或数据模式不合法")
        if type(horizon) is not int or not 1 <= horizon <= 10:
            raise ValueError("预测长度须为 1 至 10 个交易日")
        kind = payload.get("kind", "analysis")
        if kind not in ("analysis", "backtest"):
            raise ValueError("不支持的任务类型")
        if kind == "backtest" and mode != "offline":
            raise ValueError("回测只使用本地历史行情")
        with self.lock:
            if self.active_job:
                raise RuntimeError("已有任务正在运行，请等待完成")
            job_id = datetime.now(CHINA).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
            job = {"id": job_id, "status": "running", "ticker": ticker, "kind": kind,
                   "model": model, "message": "正在运行回测…" if kind == "backtest" else "正在生成分析报告…"}
            self.jobs[job_id] = job
            self.active_job = job_id
        threading.Thread(target=self._run, args=(job_id, payload), daemon=True).start()
        return dict(job)

    def _run(self, job_id, payload):
        try:
            ticker = payload["ticker"]
            folder = self.report_dir / "web" / "runs" / job_id
            folder.mkdir(parents=True, exist_ok=True)
            price_path = self.cache(ticker)
            for kind in ("price", "sentiment"):
                path = price_path if kind == "price" else price_path.with_name(f"{ticker}_sentiment.csv")
                if path.is_file():
                    shutil.copy2(path, folder / path.name)
            common = ["--model", payload["model"], "--horizon", str(payload.get("horizon", 3)), "--output-dir", str(folder)]
            if payload.get("kind", "analysis") == "backtest":
                command = [sys.executable, str(ROOT / "backtest.py"), "--price-csv", str(folder / f"{ticker}_price.csv"), *common]
            else:
                command = [sys.executable, str(ROOT / "analyze_ticker.py"), "--ticker", ticker, "--cache-dir", str(folder), *common]
                if payload["mode"] == "offline":
                    history = prices(folder / f"{ticker}_price.csv")
                    if not history:
                        raise ValueError("没有可用的本地行情，请切换到联网模式获取数据")
                    command += ["--offline", "--end", history[-1]["ds"]]
                else:
                    command += ["--refresh"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=1800, env=self._environment())
            output = (result.stdout + "\n" + result.stderr).strip()
            if result.returncode:
                raise RuntimeError(output[-5000:] or "分析进程未成功完成")
            with self.lock:
                self.jobs[job_id].update(status="completed", message="回测已完成" if payload.get("kind") == "backtest" else "分析报告已生成", output=output[-5000:])
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status="failed", message=str(exc))
        finally:
            with self.lock:
                self.active_job = None
                # Keep a bounded in-memory task history.
                for old_id in list(self.jobs)[:-30]:
                    del self.jobs[old_id]

    @staticmethod
    def _environment():
        import os
        return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def trusted_host(self):
            host = self.headers.get("Host", "")
            valid = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.server.server_port == 80:
                valid.update(("127.0.0.1", "localhost"))
            if host.lower() not in valid:
                self.respond({"error": "只允许从本机访问研究看板"}, 403)
                return False
            return True

        def respond(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.trusted_host():
                return
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/dashboard":
                    self.respond(store.dashboard(query.get("ticker", ["ICBC_A"])[0], query.get("report", [None])[0]))
                elif parsed.path.startswith("/api/jobs/"):
                    with store.lock:
                        job = dict(store.jobs.get(parsed.path.rsplit("/", 1)[-1], {}))
                    self.respond(job or {"error": "任务不存在"}, 200 if job else 404)
                else:
                    files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css", "/favicon.svg": "favicon.svg"}
                    name = files.get(parsed.path)
                    if not name:
                        self.respond({"error": "页面不存在"}, 404)
                        return
                    body = (FRONTEND / name).read_bytes()
                    self.send_response(200)
                    mime = {"html": "text/html", "js": "text/javascript", "css": "text/css", "svg": "image/svg+xml"}
                    self.send_header("Content-Type", mime[name.rsplit(".", 1)[-1]] + "; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
                    self.end_headers()
                    self.wfile.write(body)
            except ValueError as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception:
                self.respond({"error": "读取数据失败，请检查本地报告文件"}, 500)

        def do_POST(self):
            if not self.trusted_host():
                return
            if self.path != "/api/jobs":
                self.respond({"error": "接口不存在"}, 404)
                return
            if not secrets.compare_digest(self.headers.get("X-Research-Token", ""), store.token):
                self.respond({"error": "请刷新页面后重试"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("请求内容不合法")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("请求内容须为对象")
                self.respond(store.start(payload), 202)
            except (ValueError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self.respond({"error": str(exc)}, 409)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="启动工银察势研究看板")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="启动后打开浏览器")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(ResearchStore()))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"研究看板已启动：{url}\n按 Ctrl+C 关闭。运行分析使用当前 Python 环境。", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
