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

# 対応タイムフレーム
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Streamlit設定
STREAMLIT_PAGE_TITLE = "Backtest System"
STREAMLIT_LAYOUT = "wide"
