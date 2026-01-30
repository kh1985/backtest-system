"""
TradingView CSVローダー

TradingViewからエクスポートされたCSVファイルを読み込み、
OHLCVData形式に正規化する。
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import DataSource, OHLCVData, Timeframe


class TradingViewCSVLoader(DataSource):
    """TradingView CSV読み込みクラス"""

    # 出来高カラムの候補名（優先順）
    VOLUME_COLUMNS = ["volume", "Volume", "VOL", "vol"]

    def load(
        self,
        csv_path: str,
        symbol: str = "UNKNOWN",
        timeframe: Timeframe = Timeframe.M1,
        **kwargs,
    ) -> OHLCVData:
        """
        TradingView CSVを読み込んでOHLCVDataを返す

        Args:
            csv_path: CSVファイルのパス
            symbol: シンボル名
            timeframe: タイムフレーム

        Returns:
            OHLCVData
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # タイムスタンプ変換
        df = self._parse_datetime(df)

        # OHLCV正規化
        df = self._normalize_ohlcv(df)

        # datetime順にソート
        df = df.sort_values("datetime").reset_index(drop=True)

        return OHLCVData(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source="csv",
        )

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """タイムスタンプカラムをdatetimeに変換"""
        if "time" in df.columns:
            # Unix秒の場合
            if df["time"].dtype in ("int64", "float64"):
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            else:
                df["datetime"] = pd.to_datetime(df["time"])
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
        else:
            raise ValueError(
                "CSVにtime/datetime/dateカラムが見つかりません"
            )
        return df

    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCVカラムを正規化"""
        # OHLC（必須）
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"必須カラム '{col}' が見つかりません")

        # Volume（複数候補から検索）
        volume_col = None
        for candidate in self.VOLUME_COLUMNS:
            if candidate in df.columns:
                volume_col = candidate
                break

        if volume_col and volume_col != "volume":
            df["volume"] = df[volume_col]
        elif volume_col is None:
            # 出来高カラムが無い場合は0で埋める
            df["volume"] = 0.0

        # 必要カラムを抽出（追加カラムも保持）
        base_cols = ["datetime", "open", "high", "low", "close", "volume"]
        extra_cols = [c for c in df.columns if c not in base_cols and c != "time"]
        df = df[base_cols + extra_cols].copy()

        return df

    def detect_symbol_from_filename(self, csv_path: str) -> str:
        """ファイル名からシンボル名を推定"""
        name = Path(csv_path).stem
        # "BINANCE_WLDUSDT.P, 1" -> "WLDUSDT.P"
        if "_" in name:
            parts = name.split("_", 1)
            symbol_part = parts[1]
            if "," in symbol_part:
                symbol_part = symbol_part.split(",")[0].strip()
            return symbol_part
        return name

    def detect_timeframe_from_filename(self, csv_path: str) -> Optional[Timeframe]:
        """ファイル名からタイムフレームを推定"""
        name = Path(csv_path).stem
        # "BINANCE_WLDUSDT.P, 15" -> 15分
        if "," in name:
            tf_str = name.split(",")[-1].strip()
            tf_mapping = {
                "1": Timeframe.M1,
                "5": Timeframe.M5,
                "15": Timeframe.M15,
                "60": Timeframe.H1,
                "240": Timeframe.H4,
                "D": Timeframe.D1,
                "1D": Timeframe.D1,
            }
            return tf_mapping.get(tf_str)
        return None
