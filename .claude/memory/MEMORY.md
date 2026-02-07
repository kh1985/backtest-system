# Prism プロジェクトメモリ

## 最重要: 検証方法の教訓（2026-02-07）

### 固定OOSは最終判定に使うな
- Train60/Val20/Test20の固定分割はTest区間の相場環境に結果が左右される
- 特にレジーム別戦略では、Test区間のdowntrend/uptrend割合がPnLを直接支配
- **実例**: 「6/10銘柄PASS、エッジ確定！」→ WFAで「1/10 PASS」に逆転
- 原因: 2025年Test区間のdowntrend割合が45%（通常30%台）→ ショート戦略に追い風
- **固定OOSはスクリーニング用、最終判定はWFA必須**

### WFA (Walk-Forward Analysis) が正解
- `optimizer/walk_forward.py` に実装済み
- `scripts/modal_wfa.py` でModal並列実行可能
- Anchored WFA: ISの開始は常にデータ先頭、5フォールド
- CR (Consistency Ratio) >= 60% がPASS基準
- volume_spike_short / downtrend: WFA PASS 1/10（XRPのみ、PnLはマイナス）

## 現状（2026-02-07時点）

### 21種テンプレートのWFA検証はまだvolume_spike_shortのみ
- 他テンプレート（ma_crossover_short, rsi_reversal等）のWFAは未実施
- range系のWFAも未実施
- **次回要検討: 全テンプレートWFA or 戦略構造の見直し**

## 技術的な学び

### 過学習対策
- OOSは過学習を「検出」するだけで「防止」しない
- 探索空間削減（exit 69→3）+ OOS検出 の両方が必要
- min_oos_trades=20 で統計的信頼性担保
- **WFAでCR >= 60%が最終関門**

### パイプラインの盲点
- 全テンプレート混合OOSでは各銘柄のベスト1テンプレートしかOOS検証されない
- テンプレート固定横断テスト（`--templates`フラグ）で真の横断性が判明

### レジーム検出
- Dual-TF EMA (4h+1h) が最も安定
- EMA > SMA（反応速度）、SAR不可（range定義できない）

## 環境メモ
- Windows: `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` 必須（cp932問題）
- Modal CLIは bash環境変数形式で渡す: `PYTHONUTF8=1 modal run ...`
- Pythonパス: `C:\Users\k-hachiya\AppData\Local\Programs\Python\Python312\python.exe`

## Modalスクリプト一覧
| スクリプト | 用途 |
|-----------|------|
| `scripts/modal_fetch_data.py` | Binance → Modal Volume データDL |
| `scripts/modal_optimize.py` | バッチOOS最適化（スクリーニング用） |
| `scripts/modal_wfa.py` | WFA検証（最終判定用） |
| `scripts/modal_regime_check.py` | レジーム分布確認（Train/Val/Test区間別） |
| `scripts/modal_download.py` | 結果DL |

## ファイル構成
- セッションログ: `.claude/memory/sessions/YYYY-MM-DD.md`
- ロードマップ: `.claude/memory/roadmap.md`
