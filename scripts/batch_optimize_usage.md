# batch_optimize.py 使い方

## 概要
30銘柄×2期間（2024/2025）のデータに対し、14テンプレート×パラメータグリッドの自動最適化を実行し、レジーム別コンセンサス戦略の抽出とクロスバリデーションを行う。

## 実行方法

```bash
cd /path/to/backtest-system

# 全銘柄×全期間（30銘柄×2期間 ≒ 約13分）
python scripts/batch_optimize.py

# 特定銘柄のみ
python scripts/batch_optimize.py --symbols BTCUSDT,ETHUSDT

# 特定期間のみ
python scripts/batch_optimize.py --periods 20250201-20260130

# ワーカー数指定（デフォルト4）
python scripts/batch_optimize.py --workers 8

# クロスバリデーション省略
python scripts/batch_optimize.py --skip-cv
```

## 設定変更
スクリプト冒頭の定数を編集:

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `EXEC_TF` | `"15m"` | 実行タイムフレーム |
| `HTF` | `"1h"` | 上位タイムフレーム |
| `TREND_METHOD` | `"ma_cross"` | トレンド検出（`ma_cross` / `adx` / `combined`） |
| `MA_FAST` | `20` | MA短期期間 |
| `MA_SLOW` | `50` | MA長期期間 |
| `N_WORKERS` | `4` | 並列ワーカー数 |
| `INITIAL_CAPITAL` | `10000.0` | 初期資金 |
| `COMMISSION_PCT` | `0.04` | 手数料% |
| `CONSENSUS_TOP_N` | `5` | コンセンサス集計時の各銘柄上位N件 |
| `CV_TOP_N` | `3` | CV対象の各レジーム上位N戦略 |

## 処理フロー

1. **Phase 1**: 全銘柄×全期間のグリッドサーチ（1銘柄 ≒ 13秒）
2. **Phase 2**: レジーム別コンセンサス分析（テンプレート頻度・パラメータ収束度）
3. **Phase 3**: クロスバリデーション（2024ベスト → 2025検証）

## 出力ファイル

```
results/batch/YYYYMMDD_HHMMSS/
├── config.json               # 実行時の設定
├── optimization/             # 個別最適化結果（銘柄×期間）
│   ├── BTCUSDT_20240201-20250131.json
│   ├── BTCUSDT_20250201-20260130.json
│   └── ...
├── consensus.json            # レジーム別コンセンサス戦略
├── cross_validation.json     # 2024→2025のクロスバリデーション
└── report.txt                # テキストレポート（人間可読）
```

## レポートの読み方

### コンセンサス戦略
- **出現率**: 全銘柄×期間のうち、その戦略がTOP-Nに入った割合
- **PnL中央値/WR/PF**: 出現した全ケースの中央値
- 出現率が高い＝多銘柄で安定して有効な戦略

### クロスバリデーション
- **一貫性**: 2024でプラス かつ 2025でもプラスの銘柄数
- **PnL変化率**: 2024→2025でのPnL変化（マイナスが大きい＝過学習の疑い）

## データ前提
- `inputdata/` に `{SYMBOL}-{EXEC_TF}-{PERIOD}-merged.csv` と `{SYMBOL}-{HTF}-{PERIOD}-merged.csv` のペアが必要
- 期間: `20240201-20250131`（29銘柄）、`20250201-20260130`（30銘柄）
