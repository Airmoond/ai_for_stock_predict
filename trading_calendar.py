"""Exchange sessions including holidays; never silently substitute weekdays."""
import pandas as pd


class CalendarError(RuntimeError):
    pass


def calendar_name(symbol):
    if symbol.endswith((".SS", ".SZ")):
        return "XSHG"
    if symbol.endswith(".HK"):
        return "XHKG"
    if "." not in symbol:
        return "XNYS"
    raise CalendarError(f"没有配置 {symbol} 的交易日历")


def _calendar(symbol, start, end):
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise CalendarError("缺少交易日历依赖，请安装 requirements.txt") from exc
    try:
        return xcals.get_calendar(calendar_name(symbol), start=start, end=end)
    except (ValueError, KeyError) as exc:
        raise CalendarError(f"交易日历不覆盖 {start} 至 {end}，请更新 exchange-calendars: {exc}") from exc


def _dates(labels):
    labels = pd.DatetimeIndex(labels)
    return labels.tz_localize(None).normalize() if labels.tz is not None else labels.normalize()


def sessions_in_range(symbol, start, end):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start > end:
        return pd.DatetimeIndex([])
    calendar = _calendar(symbol, start, end)
    labels = _dates(calendar.sessions)
    return labels[(labels >= start) & (labels <= end)]


def latest_completed_session(symbol, end=None, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now 必须包含时区")
    local_zone = "Asia/Shanghai" if symbol.endswith((".SS", ".SZ", ".HK")) else "America/New_York"
    today = now.tz_convert(local_zone).tz_localize(None).normalize()
    cutoff = min(today, pd.Timestamp(end).normalize()) if end else today
    calendar = _calendar(symbol, cutoff - pd.Timedelta(days=40), cutoff)
    completed = [day for day in calendar.sessions
                 if calendar.session_close(day) + pd.Timedelta(minutes=15) <= now]
    if not completed:
        raise CalendarError("指定日期之前没有已收盘的交易日")
    return _dates(completed)[-1]


def next_sessions(symbol, after, horizon):
    if horizon < 1:
        raise ValueError("预测长度必须大于零")
    after = pd.Timestamp(after).normalize()
    # The library's default end respects the last year of its holiday data.
    # Asking for a padded end in January would otherwise reject valid December dates.
    calendar = _calendar(symbol, after - pd.Timedelta(days=40), None)
    days = _dates(calendar.sessions)
    days = days[days > after]
    if len(days) < horizon:
        raise CalendarError("交易日历未覆盖足够的未来交易日，请更新 exchange-calendars")
    return days[:horizon]
