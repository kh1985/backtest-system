import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from strategy.conditions import TimeBasedCondition

# テストデータ作成（UTC時刻）
test_data = pd.DataFrame({
    'datetime': pd.date_range('2025-02-01 00:00', periods=24, freq='H')  # UTC
})

# TimeBasedCondition作成（アジア時間: 1:00-14:59 JST = UTC 16:00-5:59）
cond = TimeBasedCondition(start_hour=1, end_hour=15)

# 各行でevaluate
for idx, row in test_data.iterrows():
    result = cond.evaluate(row)
    jst_time = row['datetime'] + pd.Timedelta(hours=9)
    print(f"UTC: {row['datetime']}, JST: {jst_time}, Result: {result}")
