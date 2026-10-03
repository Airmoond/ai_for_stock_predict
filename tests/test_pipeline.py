import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import analyze_ticker
import backtest
import fetch_price
import fetch_sentiment
import timesfm_model
from ensemble import combine_signals, sentiment_details
from indicators import rsi, technical_score, technical_snapshot
from recommend import make_recommendation
from storage import write_json
from trading_calendar import CalendarError, latest_completed_session, next_sessions


def prices(count=60):
    return pd.DataFrame({"ds": pd.bdate_range("2025-06-02", periods=count),
                         "Close": np.linspace(5, 6, count), "volume": 100})


@pytest.mark.parametrize("after,expected", [
    ("2025-10-10", ["2025-10-13", "2025-10-14", "2025-10-15"]),
    ("2025-09-30", ["2025-10-09", "2025-10-10", "2025-10-13"]),
    ("2026-09-30", ["2026-10-08", "2026-10-09", "2026-10-12"]),
])
def test_shanghai_weekend_and_holidays(after, expected):
    assert list(next_sessions("601398.SS", after, 3).strftime("%Y-%m-%d")) == expected


def test_hong_kong_holidays():
    dates = next_sessions("1398.HK", "2025-12-24", 2)
    assert list(dates.strftime("%Y-%m-%d")) == ["2025-12-29", "2025-12-30"]


@pytest.mark.parametrize("now,expected", [
    ("2025-10-10T14:00:00+08:00", "2025-10-09"),
    ("2025-10-10T15:30:00+08:00", "2025-10-10"),
    ("2026-10-03T12:00:00+08:00", "2026-09-30"),
])
def test_only_completed_sessions(now, expected):
    assert str(latest_completed_session("601398.SS", now=now).date()) == expected


def test_offline_cache_is_filtered_and_marked_stale(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"ds": ["2024-01-02", "2025-10-09", "2025-10-10"], "Close": [3, 7, 7.1]}).to_csv(path, index=False)
    df = fetch_price.fetch_price_to_csv("601398.SS", "2025-10-09", None, path,
                                      offline=True, now="2025-10-13T18:00:00+08:00")
    assert len(df) == 2
    assert df.attrs["quality"]["stale"]
    assert df.attrs["quality"]["expected_as_of"] == "2025-10-13"


def test_outdated_cache_refreshes(tmp_path, monkeypatch):
    path = tmp_path / "prices.csv"
    old = pd.DataFrame({"ds": ["2025-10-09"], "Close": [7]})
    old.to_csv(path, index=False)
    new = pd.DataFrame({"ds": ["2025-10-09", "2025-10-10"], "Close": [6.9, 7.1]})
    monkeypatch.setattr(fetch_price, "_download_prices", lambda *args: new)
    df = fetch_price.fetch_price_to_csv("601398.SS", "2025-10-09", None, path, now="2025-10-10T18:00:00+08:00")
    assert df.Close.iloc[0] == 6.9  # Old adjusted prices are replaced, not appended.
    assert df.attrs["quality"]["source"] == "download"
    assert not df.attrs["quality"]["stale"]


def test_complete_fresh_cache_avoids_network(tmp_path, monkeypatch):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"ds": ["2025-10-09", "2025-10-10"], "Close": [7, 7.1]}).to_csv(path, index=False)
    def unexpected_download(*args):
        pytest.fail("完整的新缓存不应访问网络")
    monkeypatch.setattr(fetch_price, "_download_prices", unexpected_download)
    df = fetch_price.fetch_price_to_csv("601398.SS", "2025-10-09", "2025-10-10", path,
                                      now="2025-10-10T18:00:00+08:00")
    assert df.attrs["quality"]["source"] == "cache"


def test_yahoo_inclusive_end_and_adjustment(monkeypatch):
    seen = {}
    def download(symbol, **kwargs):
        seen.update(kwargs)
        return prices()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    fetch_price._download_prices("1398.HK", "2025-10-09", "2025-10-10")
    assert seen["end"] == "2025-10-11"
    assert seen["auto_adjust"]


def test_failed_refresh_preserves_cache_and_strict_mode_raises(tmp_path, monkeypatch):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"ds": ["2025-10-09"], "Close": [7]}).to_csv(path, index=False)
    original = path.read_bytes()
    def fail(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr(fetch_price, "_download_prices", fail)
    args = ("601398.SS", "2025-10-09", None, path)
    df = fetch_price.fetch_price_to_csv(*args, now="2025-10-10T18:00:00+08:00")
    assert df.attrs["quality"]["source"] == "cache_fallback"
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError):
        fetch_price.fetch_price_to_csv(*args, allow_stale=False, now="2025-10-10T18:00:00+08:00")


def test_yahoo_multilevel_columns():
    frame = pd.DataFrame([[7, 100]], index=pd.DatetimeIndex(["2025-10-10"], name="Date"),
                         columns=pd.MultiIndex.from_tuples([("Close", "1398.HK"), ("Volume", "1398.HK")]))
    df = fetch_price.normalize_prices(frame)
    assert df.Close.iloc[0] == 7
    assert df.volume.iloc[0] == 100


def test_price_cleanup_removes_invalid_values_and_duplicate_dates():
    frame = pd.DataFrame({"ds": ["2025-01-02", "bad", "2025-01-03", "2025-01-02"],
                          "Close": [3, 4, np.inf, 5]})
    assert fetch_price.normalize_prices(frame).Close.tolist() == [5]


def test_rss_xml_description_and_keyword_case(monkeypatch):
    xml = b'<rss><channel><item><title>icbc &amp; profit</title><pubDate>Fri, 10 Oct 2025 10:00:00 GMT</pubDate><description>&lt;b&gt;Growth&lt;/b&gt;</description></item></channel></rss>'
    response = SimpleNamespace(content=xml, raise_for_status=lambda: None)
    monkeypatch.setattr(fetch_sentiment.requests, "get", lambda *args, **kwargs: response)
    items = fetch_sentiment.fetch_news_items()
    assert items[0]["title"] == "icbc & profit"
    assert items[0]["description"] == "Growth"


def test_news_duplicates_bad_dates_and_future_items_are_excluded(monkeypatch):
    monkeypatch.setattr(fetch_sentiment, "score_sentiment", lambda text: 0.2)
    valid = {"title": "工行利润下降", "pubDate": "2025-10-10T08:00:00Z"}
    news = [valid, valid.copy(), {"title": "无日期", "pubDate": None},
            {"title": "未来", "pubDate": "2025-10-11T08:00:00Z"}]
    df = fetch_sentiment.daily_sentiment_aggregate(news, as_of="2025-10-10")
    assert len(df) == 1
    assert df.volume.iloc[0] == 1
    assert df.sentiment_index.iloc[0] < 0
    assert df.attrs["warnings"]


def test_volume_amplifies_negative_sentiment_without_flipping_sign():
    assert -1 <= fetch_sentiment.sentiment_index(0.1, 300) < fetch_sentiment.sentiment_index(0.1, 1) < 0
    assert fetch_sentiment.sentiment_index(0.5, 300) == 0
    assert 0 < fetch_sentiment.sentiment_index(0.9, 300) <= 1


def test_language_routing(monkeypatch):
    monkeypatch.setattr(fetch_sentiment, "score_sentiment_zh", lambda text: 0.2)
    monkeypatch.setattr(fetch_sentiment, "_english_analyzer", lambda: SimpleNamespace(polarity_scores=lambda text: {"compound": 0.8}))
    assert fetch_sentiment.score_sentiment("工行利润下跌") == 0.2
    assert fetch_sentiment.score_sentiment("ICBC profit rises") == 0.9


def test_actual_english_sentiment_direction():
    assert fetch_sentiment.score_sentiment("ICBC reports excellent growth and strong profits") > 0.5
    assert fetch_sentiment.score_sentiment("ICBC suffers terrible losses and fraud scandal") < 0.5


def test_news_failure_does_not_erase_cache(tmp_path, monkeypatch):
    path = tmp_path / "sentiment.csv"
    pd.DataFrame({"ds": ["2025-10-10"], "sentiment_mean": [0.2], "volume": [3], "sentiment_index": [0.9]}).to_csv(path, index=False)
    original = path.read_bytes()
    def fail():
        raise RuntimeError("RSS unavailable")
    monkeypatch.setattr(fetch_sentiment, "fetch_news_items", fail)
    df = fetch_sentiment.build_sentiment_index_csv(path, as_of="2025-10-10")
    assert df.sentiment_index.iloc[0] < 0  # Legacy index recomputed from raw fields.
    assert path.read_bytes() == original
    assert df.attrs["quality"]["warnings"]


def test_first_news_refresh_creates_signed_cache(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "sentiment.csv"
    monkeypatch.setattr(fetch_sentiment, "fetch_news_items", lambda: [
        {"title": "ICBC reports losses", "pubDate": "2025-10-10T08:00:00Z"}])
    monkeypatch.setattr(fetch_sentiment, "score_sentiment", lambda text: 0.1)
    df = fetch_sentiment.build_sentiment_index_csv(path, as_of="2025-10-10")
    assert path.exists()
    assert df.attrs["quality"]["source"] == "rss"
    assert pd.read_csv(path).sentiment_index.iloc[0] < 0


def test_sentiment_shock_compares_with_prior_changes():
    means = [0.55, 0.56, 0.57, 0.565, 0.58, 0.59, 0.585, 0.1]
    df = pd.DataFrame({"ds": pd.date_range("2025-10-03", periods=len(means)),
                       "sentiment_mean": means, "volume": 1})
    assert sentiment_details(df, pd.Timestamp("2025-10-10"))["shock"]


def test_one_day_sentiment_is_finite_and_future_news_excluded():
    df = pd.DataFrame({"ds": ["2025-10-10", "2025-10-11"], "sentiment_mean": [0.2, 1], "volume": [1, 300]})
    details = sentiment_details(df, pd.Timestamp("2025-10-10"))
    assert np.isfinite(details["score"]) and details["score"] < 0
    assert details["available"]
    assert not sentiment_details(df.iloc[:1], pd.Timestamp("2025-10-20"))["available"]


@pytest.mark.parametrize("values,expected", [(np.arange(1, 31), 100), (np.arange(30, 0, -1), 0), (np.ones(30), 50)])
def test_rsi_edge_cases(values, expected):
    assert rsi(pd.Series(values)).iloc[-1] == expected


def test_flat_prices_are_neutral():
    df = prices()
    df["Close"] = 5
    assert technical_score(df).iloc[-1] == 0


def test_volume_golden_cross_condition_uses_actual_volume():
    df = prices()
    df["Close"] = [10] * 50 + [9] * 9 + [15]
    df.loc[df.index[-1], "volume"] = 200
    assert technical_snapshot(df)["volume_surge_and_golden_cross"]
    df.loc[df.index[-1], "volume"] = 100
    assert not technical_snapshot(df)["volume_surge_and_golden_cross"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_forecast_is_rejected(value):
    df = prices()
    forecast = pd.DataFrame({"ds": [df.ds.iloc[-1] + pd.Timedelta(days=1)], "timesfm": [value]})
    with pytest.raises(ValueError):
        combine_signals(df, forecast, None, technical_score(df))


def test_recommendation_is_grounded_and_stale_data_blocks_rating():
    metrics = {"combo_score": 0.5, "exp_return": 0.03, "senti_score": -0.5,
               "tech_score": -0.4, "sentiment_available": True}
    rec = make_recommendation(metrics)
    assert "共振" not in rec["note"] and "概率" not in rec["note"]
    assert "-0.500" in rec["note"]
    assert make_recommendation(metrics, data_quality={"stale": True})["rating"] == "数据不足"
    assert make_recommendation(metrics, technical={"below_sma30_and_oversold": True})["rating"] == "回避/减仓"
    assert make_recommendation({**metrics, "sentiment_shock": True})["rating"] == "谨慎增持/持有"


def test_timesfm_adapter_truncates_context_and_assigns_exchange_sessions(monkeypatch):
    df = prices(600)
    df["ds"] = pd.bdate_range(end="2025-10-10", periods=600)
    seen = {}
    def forecast(*, inputs, freq):
        seen.update(length=len(inputs[0]), freq=freq)
        return np.full((1, 128), 7.0), None
    monkeypatch.setattr(timesfm_model, "_get_model", lambda *args: SimpleNamespace(forecast=forecast))
    result = timesfm_model.predict_next(df, symbol="601398.SS")
    assert seen == {"length": 512, "freq": [0]}
    assert len(result) == 3
    assert (result.ds > df.ds.iloc[-1]).all()


def test_timesfm_constructor_uses_cpu_and_mean_api(monkeypatch):
    seen = {}
    fake = SimpleNamespace(TimesFmHparams=lambda **kwargs: kwargs,
                           TimesFmCheckpoint=lambda **kwargs: kwargs,
                           TimesFm=lambda **kwargs: seen.update(kwargs) or SimpleNamespace())
    monkeypatch.setitem(sys.modules, "timesfm", fake)
    monkeypatch.setattr(timesfm_model, "sys", SimpleNamespace(version_info=(3, 11)))
    timesfm_model._get_model.cache_clear()
    try:
        timesfm_model._get_model(128, "cpu")
        assert seen["hparams"]["backend"] == "cpu"
        assert seen["hparams"]["point_forecast_mode"] == "mean"
    finally:
        timesfm_model._get_model.cache_clear()


def test_backtest_causal_inputs_delayed_execution_and_costs(monkeypatch):
    df = prices(40)
    seen = []
    def forecast(history, horizon, **kwargs):
        seen.append(history.ds.iloc[-1])
        return np.repeat(history.Close.iloc[-1] * 1.1, horizon)
    monkeypatch.setattr(backtest, "forecast_prices", forecast)
    monkeypatch.setattr(backtest, "make_recommendation", lambda *args, **kwargs: {"rating": "买入"})
    summary, detail = backtest.walk_forward(df, min_train=30, test_sessions=5, cost_bps=10)
    assert list(pd.to_datetime(detail.signal_date)) == seen
    assert (pd.to_datetime(detail.execution_date) > pd.to_datetime(detail.signal_date)).all()
    assert (pd.to_datetime(detail.return_date) > pd.to_datetime(detail.execution_date)).all()
    enter = df.set_index("ds").loc[pd.Timestamp(detail.execution_date.iloc[0]), "Close"]
    exit_price = df.set_index("ds").loc[pd.Timestamp(detail.return_date.iloc[-1]), "Close"]
    expected_return = exit_price / enter * (1 - 0.001) ** 2 - 1
    assert summary["strategy"]["turnover"] == 2
    assert summary["strategy"]["total_return"] == pytest.approx(expected_return)
    assert summary["buy_and_hold"]["total_return"] == pytest.approx(expected_return)


def test_naive_direction_ignores_float_roundoff():
    result = backtest.forecast_metrics([7.2, 7.4], [7.3 + 1e-15, 7.3 - 1e-15], [7.3, 7.3])
    assert result["direction_accuracy"] == 0


def test_calendar_year_boundary_and_missing_future_coverage():
    assert list(next_sessions("601398.SS", "2026-12-28", 3).strftime("%Y-%m-%d")) == ["2026-12-29", "2026-12-30", "2026-12-31"]
    with pytest.raises(CalendarError):
        next_sessions("601398.SS", "2026-12-31", 3)


def test_backtest_early_signals_are_unchanged_by_later_prices():
    df = prices(50)
    _, first = backtest.walk_forward(df, model="naive", min_train=30, test_sessions=100)
    changed = df.copy()
    changed.loc[changed.index[-1], "Close"] *= 2
    _, second = backtest.walk_forward(changed, model="naive", min_train=30, test_sessions=100)
    pd.testing.assert_frame_equal(first.iloc[:-3], second.iloc[:-3])


def test_json_write_rejects_nan_without_overwriting(tmp_path):
    path = tmp_path / "report.json"
    write_json({"ok": True}, path)
    with pytest.raises(ValueError):
        write_json({"bad": float("nan")}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_offline_analysis_end_to_end_preserves_historical_report(tmp_path):
    root = analyze_ticker.REPORT_DIR
    historical = (root / "ICBC_A_report.json").read_bytes()
    result = analyze_ticker.analyze("ICBC_A", "601398.SS", end="2025-10-10", offline=True,
                                    model="naive", cache_dir=root, output_dir=tmp_path)
    saved = json.loads((tmp_path / "ICBC_A_naive_report.json").read_text(encoding="utf-8"))
    assert saved["data_as_of"] == "2025-10-10"
    assert saved["forecast"][0]["ds"] == "2025-10-13"
    assert saved["model"]["name"] == "naive"
    assert not saved["data_quality"]["stale"]
    assert saved["metrics"] == result["metrics"]
    assert (root / "ICBC_A_report.json").read_bytes() == historical
