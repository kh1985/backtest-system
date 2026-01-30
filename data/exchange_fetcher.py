"""
取引所データフェッチャー

ccxtを使用して取引所からOHLCVデータを取得する。
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from .base import DataSource, OHLCVData, Timeframe


class ExchangeFetcher(DataSource):
    """ccxt経由で取引所からデータを取得"""

    def __init__(
        self,
        exchange_id: str = "mexc",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is not installed. Run: pip install ccxt"
            )

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")

        config = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret

        self.exchange = exchange_class(config)
        self.exchange_id = exchange_id

    def load(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[str] = None,
        limit: int = 1000,
        **kwargs,
    ) -> OHLCVData:
        """
        取引所からOHLCVデータを取得

        Args:
            symbol: シンボル（例: "WLD/USDT"）
            timeframe: タイムフレーム
            since: 開始日時（ISO形式文字列、例: "2026-01-01T00:00:00Z"）
            limit: 取得する本数（最大）

        Returns:
            OHLCVData
        """
        since_ms = None
        if since:
            since_ms = self.exchange.parse8601(since)

        ohlcv = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe.value,
            since=since_ms,
            limit=limit,
        )

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop(columns=["timestamp"])
        df = df.sort_values("datetime").reset_index(drop=True)

        return OHLCVData(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source=f"exchange:{self.exchange_id}",
        )

    def fetch_all(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> OHLCVData:
        """
        ページネーションで全期間のデータを取得

        Args:
            symbol: シンボル
            timeframe: タイムフレーム
            start_date: 開始日（ISO形式）
            end_date: 終了日（ISO形式、Noneなら現在まで）

        Returns:
            OHLCVData
        """
        all_data = []
        since_ms = self.exchange.parse8601(start_date)
        end_ms = (
            self.exchange.parse8601(end_date)
            if end_date
            else int(datetime.now().timestamp() * 1000)
        )

        tf_ms = timeframe.to_minutes() * 60 * 1000

        while since_ms < end_ms:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe.value,
                since=since_ms,
                limit=1000,
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)
            last_ts = ohlcv[-1][0]

            # 進捗が無ければ終了
            if last_ts <= since_ms:
                break

            since_ms = last_ts + tf_ms

        if not all_data:
            df = pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
        else:
            df = pd.DataFrame(
                all_data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ],
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop(columns=["timestamp"])
            df = df.drop_duplicates(subset=["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            # 終了日でフィルター
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df["datetime"] <= end_dt].reset_index(drop=True)

        return OHLCVData(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source=f"exchange:{self.exchange_id}",
        )
