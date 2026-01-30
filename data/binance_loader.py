"""
Binance Data CSV/ZIPローダー

data.binance.vision からダウンロードしたCSV/ZIPを読み込み、
OHLCVData形式に正規化する。

ファイル名パターン: WLDUSDT-1m-2025-01.csv (.zip)
カラム: open_time, open, high, low, close, volume,
        close_time, quote_volume, count, taker_buy_volume,
        taker_buy_quote_volume, ignore
"""

import zipfile
import tempfile
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import DataSource, OHLCVData, Timeframe


# Binance CSVのカラム名（ヘッダーなし形式）
BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


class BinanceCSVLoader(DataSource):
    """Binance Data (data.binance.vision) CSV/ZIP読み込み"""

    def load(
        self,
        file_path: str,
        symbol: str = "",
        timeframe: Timeframe = Timeframe.M1,
        **kwargs,
    ) -> OHLCVData:
        """
        単一のCSV/ZIPファイルを読み込む

        Args:
            file_path: CSV or ZIP ファイルパス
            symbol: シンボル名（空の場合はファイル名から自動検出）
            timeframe: タイムフレーム（自動検出も可能）
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # ファイル名から自動検出
        if not symbol:
            symbol = self.detect_symbol(path.name)
        detected_tf = self.detect_timeframe(path.name)
        if detected_tf:
            timeframe = detected_tf

        # ZIP or CSV
        if path.suffix.lower() == ".zip":
            df = self._load_zip(path)
        else:
            df = self._load_csv(path)

        df = self._normalize(df)
        df = df.sort_values("datetime").reset_index(drop=True)

        return OHLCVData(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source="binance_csv",
        )

    def load_multiple(
        self,
        file_paths: List[str],
        symbol: str = "",
        timeframe: Timeframe = Timeframe.M1,
    ) -> OHLCVData:
        """
        複数CSVを結合して読み込む（月別ファイル等）

        Args:
            file_paths: CSVファイルパスのリスト
            symbol: シンボル名
            timeframe: タイムフレーム
        """
        dfs = []
        for fp in file_paths:
            ohlcv = self.load(fp, symbol=symbol, timeframe=timeframe)
            dfs.append(ohlcv.df)
            if not symbol:
                symbol = ohlcv.symbol
            timeframe = ohlcv.timeframe

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"])
        combined = combined.sort_values("datetime").reset_index(drop=True)

        return OHLCVData(
            df=combined,
            symbol=symbol,
            timeframe=timeframe,
            source="binance_csv",
        )

    def _load_zip(self, path: Path) -> pd.DataFrame:
        """ZIPファイルからCSVを読み込む"""
        with zipfile.ZipFile(path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found in ZIP: {path}")

            dfs = []
            for csv_name in csv_names:
                with zf.open(csv_name) as f:
                    df = self._read_csv_content(f)
                    dfs.append(df)

            return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """CSVファイルを読み込む"""
        return self._read_csv_content(path)

    def _read_csv_content(self, source) -> pd.DataFrame:
        """CSVコンテンツを読み込み（ヘッダー有無を自動判定）"""
        df = pd.read_csv(source, header=None)

        # ヘッダー行があるか判定
        first_val = str(df.iloc[0, 0])
        if first_val.replace("_", "").isalpha():
            # ヘッダー付き
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        else:
            # ヘッダーなし（Binance標準）
            if len(df.columns) >= len(BINANCE_COLUMNS):
                df.columns = BINANCE_COLUMNS[:len(df.columns)]
            elif len(df.columns) >= 6:
                # 最低限OHLCVがある
                cols = BINANCE_COLUMNS[:len(df.columns)]
                df.columns = cols

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binance形式をOHLCV標準形式に変換"""
        result = pd.DataFrame()

        # open_timeをdatetimeに変換
        if "open_time" in df.columns:
            ts = pd.to_numeric(df["open_time"], errors="coerce")
            # マイクロ秒 / ミリ秒 / 秒の判定
            first_val = ts.iloc[0]
            if first_val > 1e15:
                result["datetime"] = pd.to_datetime(ts, unit="us")
            elif first_val > 1e12:
                result["datetime"] = pd.to_datetime(ts, unit="ms")
            else:
                result["datetime"] = pd.to_datetime(ts, unit="s")
        elif "datetime" in df.columns:
            result["datetime"] = pd.to_datetime(df["datetime"])
        else:
            raise ValueError("open_time or datetime column not found")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                raise ValueError(f"Required column '{col}' not found")

        return result

    @staticmethod
    def detect_symbol(filename: str) -> str:
        """ファイル名からシンボルを検出

        例: WLDUSDT-1m-2025-01.csv → WLDUSDT
            BTCUSDT-15m-2024-12-01.zip → BTCUSDT
        """
        stem = Path(filename).stem
        parts = stem.split("-")
        if parts:
            return parts[0]
        return "UNKNOWN"

    @staticmethod
    def detect_timeframe(filename: str) -> Optional[Timeframe]:
        """ファイル名からタイムフレームを検出

        例: WLDUSDT-1m-2025-01.csv → Timeframe.M1
        """
        stem = Path(filename).stem
        parts = stem.split("-")
        if len(parts) >= 2:
            tf_str = parts[1]
            tf_mapping = {
                "1m": Timeframe.M1,
                "5m": Timeframe.M5,
                "15m": Timeframe.M15,
                "1h": Timeframe.H1,
                "4h": Timeframe.H4,
                "1d": Timeframe.D1,
            }
            return tf_mapping.get(tf_str)
        return None
