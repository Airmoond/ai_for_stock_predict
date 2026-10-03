"""Dashboard data integrity, local HTTP boundaries and subprocess integration."""
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from webapp import ResearchStore, make_handler, prices


@pytest.fixture
def store(tmp_path):
    (tmp_path / "ICBC_A_price.csv").write_text(
        "ds,Close,volume\n2025-10-08,7.0,100\n2025-10-09,7.2,200\n2025-10-10,7.3,150\n2025-10-13,7.5,180\n", encoding="utf-8")
    (tmp_path / "ICBC_A_sentiment.csv").write_text(
        "ds,sentiment_mean,volume,sentiment_index\n2025-10-09,0.1,3,0.7\n2025-10-13,0.9,5,0.8\n", encoding="utf-8")
    report = {"schema_version": 2, "ticker": "ICBC_A", "as_of": "2025-10-11T12:00:00+08:00",
              "data_as_of": "2025-10-10", "model": {"name": "naive"}, "warnings": [],
              "forecast": [{"ds": "2025-10-13", "price": 7.3}]}
    (tmp_path / "ICBC_A_naive_report.json").write_text(json.dumps(report), encoding="utf-8")
    return ResearchStore(tmp_path)


def test_report_snapshot_excludes_future_prices_and_news(store):
    dashboard = store.dashboard("ICBC_A")
    assert dashboard["prices"][-1]["ds"] == "2025-10-10"
    assert [r["ds"] for r in dashboard["sentiment"]] == ["2025-10-09"]
    # Legacy CSV's positive-only sentiment index must not become a bullish signal.
    assert dashboard["sentiment"][0]["index"] < 0
    assert dashboard["forecast"] == [{"ds": "2025-10-13", "price": 7.3}]


def test_legacy_forecast_is_not_presented_as_valid_trading_days(store):
    legacy = {"ticker": "ICBC_A", "as_of": "2024-01-01", "peek_forecast": [{"ds": "2024-01-06", "timesfm": 7.3}]}
    (store.report_dir / "ICBC_A_report.json").write_text(json.dumps(legacy), encoding="utf-8")
    d = store.dashboard("ICBC_A", "ICBC_A_report.json")
    assert d["legacy"] and d["forecast"] == []
    assert any("交易日校验" in w for w in d["warnings"])


def test_selects_newest_report_and_its_own_cache(store):
    folder = store.report_dir / "new"
    folder.mkdir()
    report = {"schema_version": 2, "ticker": "ICBC_A", "as_of": "2026-01-02", "data_as_of": "2026-01-01"}
    (folder / "ICBC_A_naive_report.json").write_text(json.dumps(report), encoding="utf-8")
    (folder / "ICBC_A_price.csv").write_text("ds,Close\n2026-01-01,8.1\n", encoding="utf-8")
    d = store.dashboard("ICBC_A")
    assert d["selected_report"] == "new/ICBC_A_naive_report.json"
    assert d["price_source"] == "new/ICBC_A_price.csv"
    assert d["prices"][-1]["close"] == 8.1


def test_missing_market_is_empty_and_does_not_borrow_a_share_data(store):
    d = store.dashboard("ICBC_H")
    assert d["prices"] == [] and d["report"] is None and d["backtest"] is None


def test_web_test_fixtures_are_not_loaded_as_research_artifacts(store):
    folder = store.report_dir / "web" / "tests"
    folder.mkdir(parents=True)
    (folder / "ICBC_A_naive_report.json").write_text(json.dumps({"ticker": "ICBC_A", "as_of": "2099-01-01"}))
    (folder / "ICBC_A_price.csv").write_text("ds,Close\n2099-01-01,999\n")
    assert store.dashboard("ICBC_A")["selected_report"] == "ICBC_A_naive_report.json"
    assert store.cache("ICBC_A") == store.report_dir / "ICBC_A_price.csv"


def test_price_normalization_rejects_nonfinite_and_deduplicates(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("ds,Close\n2025-01-02,5\n2025-01-01,NaN\n2025-01-03,-1\n2025-01-02,6\ninvalid,8\n2025-01-04,inf\n")
    assert [(r["ds"], r["close"]) for r in prices(path)] == [("2025-01-02", 6)]


def test_report_selection_cannot_access_arbitrary_files(store):
    with pytest.raises(ValueError):
        store.dashboard("ICBC_A", "../../config.py")


@pytest.mark.parametrize("override", [{"ticker": "invalid"}, {"model": "shell"}, {"mode": "invalid"},
                                      {"horizon": True}, {"horizon": 11}, {"kind": "delete"}])
def test_task_input_validation_precedes_execution(store, override):
    payload = {"ticker": "ICBC_A", "model": "naive", "mode": "offline", **override}
    with pytest.raises(ValueError):
        store.start(payload)
    assert store.active_job is None


def test_job_preserves_original_cache_and_uses_isolated_output(store, monkeypatch):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="完成", stderr="")
    monkeypatch.setattr("webapp.subprocess.run", run)
    store.jobs["test"] = {"status": "running"}
    store.active_job = "test"
    before = (store.report_dir / "ICBC_A_price.csv").read_bytes()
    store._run("test", {"ticker": "ICBC_A", "model": "naive", "mode": "offline", "horizon": 3})
    command = commands[0]
    assert command[command.index("--end") + 1] == "2025-10-13"
    assert "--offline" in command
    assert "web" in command[command.index("--cache-dir") + 1]
    assert (store.report_dir / "ICBC_A_price.csv").read_bytes() == before
    assert store.jobs["test"]["status"] == "completed" and store.active_job is None


def test_subprocess_failure_is_visible_and_releases_running_job(store, monkeypatch):
    monkeypatch.setattr("webapp.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="缺少 TimesFM 依赖"))
    store.jobs["test"] = {"status": "running"}
    store.active_job = "test"
    store._run("test", {"ticker": "ICBC_A", "model": "timesfm", "mode": "offline"})
    assert store.jobs["test"]["status"] == "failed"
    assert "TimesFM" in store.jobs["test"]["message"]
    assert store.active_job is None


@pytest.fixture
def http_server(store):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def request(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_http_serves_frontend_and_refuses_project_file_exposure(http_server):
    status, body = request(http_server, "GET", "/")
    assert status == 200 and "工银察势" in body.decode()
    assert request(http_server, "GET", "/../config.py")[0] == 404
    assert request(http_server, "GET", "/reports/ICBC_A_price.csv")[0] == 404


def test_http_rejects_untrusted_host_and_missing_task_token(http_server):
    assert request(http_server, "GET", "/api/dashboard", headers={"Host": "attacker.example"})[0] == 403
    assert request(http_server, "POST", "/api/jobs", body="{}")[0] == 403


def test_http_validation_errors_are_json(http_server, store):
    status, body = request(http_server, "POST", "/api/jobs", body='{"ticker":"invalid"}',
                           headers={"X-Research-Token": store.token})
    assert status == 400 and "error" in json.loads(body)
    assert request(http_server, "GET", "/api/jobs/missing")[0] == 404
