"""
マルチタイムフレームアライメント

複数タイムフレームのデータを基準タイムフレームに合わせて結合する。
ルックアヘッドバイアスを防ぐため、上位TFの値はforward-fillで展開。
"""

from typing import Dict

import pandas as pd

from .base import Timeframe


class TimeframeAligner:
    """マルチタイムフレームデータのアライメント"""

    @staticmethod
    def align(
        data_dict: Dict[str, pd.DataFrame],
        base_timeframe: str,
    ) -> pd.DataFrame:
        """
        複数タイムフレームのDataFrameを基準TFに合わせて結合

        Args:
            data_dict: {"1m": df_1m, "15m": df_15m, "4h": df_4h}
            base_timeframe: 基準タイムフレーム（最小TF）

        Returns:
            結合されたDataFrame（基準TFのインデックス）

        例:
            4h SMA_20の値は、4hバー完了後の全1mバーに同じ値が入る。
            現在進行中の4hバーには前の完了済み4hバーの値が入る。
        """
        if base_timeframe not in data_dict:
            raise ValueError(
                f"Base timeframe '{base_timeframe}' not found in data_dict"
            )

        base_df = data_dict[base_timeframe].copy()
        base_df = base_df.set_index("datetime")

        for tf_name, tf_df in data_dict.items():
            if tf_name == base_timeframe:
                continue

            tf_data = tf_df.copy()
            tf_data = tf_data.set_index("datetime")

            # OHLCV以外のカラム（インジケーター）にサフィックスを追加
            ohlcv_cols = {"open", "high", "low", "close", "volume"}
            indicator_cols = [
                c for c in tf_data.columns if c not in ohlcv_cols
            ]

            for col in indicator_cols:
                suffixed = f"{col}_{tf_name}"
                # 上位TFの値を基準TFにマージ（forward-fill）
                merged = base_df.index.to_frame(name="base_dt")
                merged[suffixed] = None

                # 上位TFの各バーの値を、次のバーが始まるまでの基準TFバーに割り当て
                for i in range(len(tf_data)):
                    val = tf_data[col].iloc[i]
                    ts = tf_data.index[i]

                    # 次の上位TFバーの開始時刻
                    if i + 1 < len(tf_data):
                        next_ts = tf_data.index[i + 1]
                    else:
                        next_ts = base_df.index[-1] + pd.Timedelta(seconds=1)

                    # 該当する基準TFバーにforward-fill
                    mask = (base_df.index >= ts) & (base_df.index < next_ts)
                    base_df.loc[mask, suffixed] = val

        base_df = base_df.reset_index()
        return base_df

    @staticmethod
    def resample(
        df: pd.DataFrame, target_tf: str
    ) -> pd.DataFrame:
        """
        下位TFから上位TFにリサンプル

        Args:
            df: 元のDataFrame（datetime, OHLCV）
            target_tf: 変換先タイムフレーム

        Returns:
            リサンプルされたDataFrame
        """
        tf_mapping = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "1h": "1h", "4h": "4h", "1d": "1D",
        }

        if target_tf not in tf_mapping:
            raise ValueError(f"Unknown timeframe: {target_tf}")

        resampled_df = df.set_index("datetime")
        freq = tf_mapping[target_tf]

        result = resampled_df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        result = result.reset_index()
        return result
