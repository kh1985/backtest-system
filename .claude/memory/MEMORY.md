# Prism プロジェクトメモリ

## 最重要: 検証方法の教訓（2026-02-07）

### 固定OOSは最終判定に使うな
- Train60/Val20/Test20の固定分割はTest区間の相場環境に結果が左右される
- 特にレジーム別戦略では、Test区間のdowntrend/uptrend割合がPnLを直接支配
- **実例**: 「6/10銘柄PASS、エッジ確定！」→ WFAで「1/10 PASS」に逆転
- **固定OOSはスクリーニング用、最終判定はWFA必須**

### WFA (Walk-Forward Analysis) が正解
- `optimizer/walk_forward.py` + `scripts/modal_wfa.py` で実行
- Anchored WFA: ISの開始は常にデータ先頭、5フォールド
- CR (Consistency Ratio) >= 60% がPASS基準

### 探索空間サイズと過学習の関係（2026-02-07 午後に実証）
- 4テンプレ(216 configs): Range 7/10 PASS
- 21テンプレ(1449 configs): Range **3/10 PASS**（大幅悪化）
- config数が多いほどISで「たまたまの勝者」が選ばれ、OOS崩壊
- **テンプレート数を増やすな。既存テンプレの精度を上げろ**
- ISフィルタ（pf>10除外等）は偽陰性リスクあり → 探索空間制約のほうが安全

## 現状（2026-02-07 午後）

### WFA検証済み（3年間×5フォールド×10銘柄）
- Run1: 4 shortテンプレ × 3レジーム → Range最強（7/10 PASS）
- Run2: 全21テンプレ × 3レジーム → 過学習でRange 3/10に低下
- **ロバストな組み合わせ**: DOT/up+33%, ADA/up+18%, XRP/range+14.5%, SOL/down+8%

### 複合条件テンプレート追加済み（未WFA検証）
- rsi_volume_short/long: RSI極値 + RVOL + キャンドル
- bb_volume_short/long: BB帯タッチ + RVOL + キャンドル
- **次回: WFA検証で効果確認**

## 技術的な学び

### 過学習対策
- OOSは過学習を「検出」するだけで「防止」しない
- 探索空間削減（exit 69→3, テンプレ21→5種）+ WFA検出 の両方が必要
- min_oos_trades=20 で統計的信頼性担保

### レジーム検出
- Dual-TF EMA (4h+1h) が最も安定

## 環境メモ
- Windows: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` 必須（cp932問題）
- Modal CLIは bash環境変数形式で渡す: `PYTHONUTF8=1 modal run ...`

## Modalスクリプト一覧
| スクリプト | 用途 |
|-----------|------|
| `scripts/modal_fetch_data.py` | Binance → Modal Volume データDL |
| `scripts/modal_optimize.py` | バッチOOS最適化（スクリーニング用） |
| `scripts/modal_wfa.py` | WFA検証（最終判定用）`--templates all` 対応 |
| `scripts/modal_regime_check.py` | レジーム分布確認 |
| `scripts/modal_download.py` | 結果DL（`wfa/`対応済み） |

## ファイル構成
- セッションログ: `.claude/memory/sessions/YYYY-MM-DD.md`
- ロードマップ: `.claude/memory/roadmap.md`
