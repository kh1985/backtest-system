"""
設定管理
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# プロジェクトルート
BASE_DIR = Path(__file__).resolve().parent.parent

# データ関連
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
CACHE_DIR = BASE_DIR / "cache"

# 取引所API設定
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "mexc")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# バックテストデフォルト設定
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION_PCT = 0.04  # 0.04%
DEFAULT_SLIPPAGE_PCT = 0.0
DEFAULT_ENTRY_ON_NEXT_OPEN = True
DEFAULT_BARS_PER_YEAR = 365 * 24 * 4  # 15m

# 対応タイムフレーム
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
TIMEFRAME_BARS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}

# Streamlit設定
STREAMLIT_PAGE_TITLE = "Prism"
STREAMLIT_LAYOUT = "wide"
