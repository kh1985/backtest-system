"""
データキャッシュ

Parquet形式でOHLCVデータをローカルにキャッシュする。
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import CACHE_DIR


class DataCache:
    """Parquetベースのデータキャッシュ"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(
        self, symbol: str, timeframe: str, start: str = "", end: str = ""
    ) -> str:
        """キャッシュキーを生成"""
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        parts = [safe_symbol, timeframe]
        if start:
            parts.append(start.replace(":", "").replace("-", ""))
        if end:
            parts.append(end.replace(":", "").replace("-", ""))
        return "_".join(parts)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def has(self, symbol: str, timeframe: str, **kwargs) -> bool:
        """キャッシュが存在するか"""
        key = self._make_key(symbol, timeframe, **kwargs)
        return self._path(key).exists()

    def get(
        self, symbol: str, timeframe: str, **kwargs
    ) -> Optional[pd.DataFrame]:
        """キャッシュからデータを取得"""
        key = self._make_key(symbol, timeframe, **kwargs)
        path = self._path(key)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def put(
        self, df: pd.DataFrame, symbol: str, timeframe: str, **kwargs
    ) -> None:
        """キャッシュにデータを保存"""
        key = self._make_key(symbol, timeframe, **kwargs)
        path = self._path(key)
        df.to_parquet(path, index=False)

    def clear(self) -> int:
        """全キャッシュを削除"""
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        return count

    def list_cached(self) -> list:
        """キャッシュファイル一覧"""
        return [f.stem for f in self.cache_dir.glob("*.parquet")]
