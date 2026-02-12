"""
ROC値を実データで確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from indicators.momentum import ROC

# データ読み込み
data_path = Path("inputdata") / "BTCUSDT-15m-20250201-20260130-merged.csv"
if not data_path.exists():
    print(f"データファイルが存在しません: {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path, names=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
print(f"データ読み込み: {len(df)} rows")
if 'timestamp' in df.columns:
    print(f"期間: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
elif 'time' in df.columns:
    print(f"期間: {df.iloc[0]['time']} to {df.iloc[-1]['time']}")

# ROC計算（20, 30, 40）
for period in [20, 30, 40]:
    roc = ROC(period=period)
    df = roc.calculate(df)
    col = f"roc_{period}"

    print(f"\n=== ROC({period}) ===")
    print(f"Range: {df[col].min():.2f} to {df[col].max():.2f}")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")

    # 閾値カウント
    count_neg = (df[col] < 0).sum()
    count_neg5 = (df[col] < -5).sum()
    print(f"ROC < 0: {count_neg} ({count_neg/len(df)*100:.1f}%)")
    print(f"ROC < -5: {count_neg5} ({count_neg5/len(df)*100:.1f}%)")

    # サンプル値
    print(f"最近10件の値:")
    print(df[[col]].tail(10))
