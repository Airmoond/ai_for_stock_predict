import time
import yfinance as yf

def fetch_with_retry(symbol, start, end, max_retries=3, retry_delay=5, pre_delay=1):
    """
    用 Ticker.history 方式获取单只股票的历史数据，如果遇到限流（异常信息里含 "rate limit"），就休眠后重试。
    
    参数：
      - symbol: 股票代码（例如 "AAPL"）。
      - start:  起始日期，格式 "YYYY-MM-DD"。
      - end:    结束日期，格式 "YYYY-MM-DD"。
      - max_retries: 最多尝试次数（默认 3 次）。
      - retry_delay: 每次收到限流信号时，等待的秒数（默认 5 秒）。
      - pre_delay:    在真正发起请求之前的预等待（默认 1 秒），帮助降低瞬时请求流量。
    返回：
      - 一旦成功获取到 DataFrame，就返回该 DataFrame；否则最后抛出异常。
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # 在每次请求前预先 sleep，降低并发触发限流的几率
            time.sleep(pre_delay)
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end)  # 单次请求，无并行
            return data

        except Exception as e:
            text = str(e).lower()
            if "rate limit" in text or "rate limited" in text:
                attempt += 1
                if attempt < max_retries:
                    print(f"第 {attempt} 次尝试被限流，等待 {retry_delay} 秒后重试…")
                    time.sleep(retry_delay)
                    continue
            # 不是限流错误或者重试次数用尽，就抛出
            raise

if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date   = "2023-01-15"

    try:
        df = fetch_with_retry(symbol, start_date, end_date)
        if df is not None and not df.empty:
            df.to_csv("aapl_stock.csv")
            print("✅ 数据已下载并保存到 aapl_stock.csv")
        else:
            print("⚠️ 下载成功，但返回了空的数据表，请确认日期区间是否包含交易日（如 2023-01-01/02 是节假日）。")
    except Exception as err:
        print(f"❌ 下载失败：{err}")
