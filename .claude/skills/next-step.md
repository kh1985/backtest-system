# /next-step - ロードマップに基づく次ステップ自動提案・実行

ロードマップを読み、最新の結果を分析し、次に試すべきパターンを提案・実行する。

## 手順

### Phase 1: 現状把握

1. `.claude/memory/roadmap.md` を読み、現在のステップと次のステップを特定
2. `results/batch/` の最新ランフォルダを特定し、`report.md` を読む
3. 最新ランの結果サマリーを表示:
   - Run ID、条件（テンプレート、銘柄数、期間）
   - OOS PASS率、推奨戦略数
   - レジーム別・テンプレート別の傾向

### Phase 2: 分析・判定

4. ロードマップの「次のステップ」セクションを読み、候補を確認
5. 以下の判定基準で次のアクションを決定:

   **判定ロジック:**
   - 前回のPASS率が30%未満 → テンプレートの改良が必要（閾値拡張/新規複合）
   - 前回のPASS率が30%以上だがPnL < 5% → 閾値微調整でPnL向上を狙う
   - 前回で3銘柄以上PASSした戦略がある → その戦略の堅牢性をさらに検証
   - 新しいテンプレートの候補がある → 次の複合テンプレートを試す

6. 具体的な提案を表示:
   ```
   ## 次ステップ提案

   **アクション**: [テンプレート追加 / 閾値拡張 / 期間拡張 / etc.]
   **テンプレート**: [具体的なテンプレート名]
   **パラメータ**: [探索する閾値の範囲]
   **期間**: [2年 / 3年]
   **銘柄**: [10銘柄]
   **推定ジョブ数**: [N jobs]

   **根拠**: [なぜこのパターンを試すのか]
   ```

### Phase 3: 実行（ユーザー承認後）

7. ユーザーに「実行しますか？」と確認
8. 承認されたら:
   a. 必要ならテンプレートを `optimizer/templates.py` に追加
   b. `python3 -m modal run scripts/modal_optimize.py` で実行
   c. `python3 -m modal run scripts/modal_download.py` で結果DL
   d. 結果を分析して表示

### Phase 4: ロードマップ更新

9. 結果をロードマップに記録（新しいStepとして追加）
10. 次のステップ候補を更新

## 設計原則（必ず守る）

- インジケーター期間は**固定**（RSI=14, BB=20/2σ, MACD=12/26/9等）
- 探索するのは**閾値のみ**（3-4段階）
- 1テンプレートあたり9-27コンボ以下
- 全テンプレート合計100コンボ以下
- Exit は `atr_compact`（3択）固定
- レジーム検出は `dual_tf_ema`（4h+1h）固定
- OOS必須（train60/val20/test20, min_trades=20）

## BB系複合が唯一の有効パターン（確定済み）

以下は検証済みで失敗確定:
- Stoch + BB → 全滅
- MACD + BB → 全滅
- RSI + MACD → 全滅
- RSI + Volume → AND条件が厳しすぎてトレード数激減

以下が機能する組み合わせ:
- **RSI + BB** → rsi_bb_reversal_short（6銘柄PASS実績）
- **BB + Volume** → bb_volume_reversal_short（5銘柄PASS実績）
- **3重複合（RSI + BB + Volume）** → 未検証、次の候補

## 引数

- 引数なし: Phase 1-2（分析と提案のみ）
- `run`: Phase 1-4（提案→承認→実行→更新まで）
- `status`: Phase 1のみ（現状把握だけ）

## Modal実行コマンド例

```bash
# データ取得（必要な場合のみ）
python3 -m modal run scripts/modal_fetch_data.py --symbols "BTC,ETH,SOL,XRP,BNB,ADA,AVAX,DOGE,DOT,TRX" --start 20230201 --end 20260130 --timeframes "15m,1h,4h"

# 最適化実行
python3 -m modal run scripts/modal_optimize.py --symbols "BTC,ETH,SOL,XRP,BNB,ADA,AVAX,DOGE,DOT,TRX" --start 20230201 --end 20260130 --templates "rsi_bb_reversal_short,bb_volume_reversal_short" --exit-profiles atr_compact --regime dual_tf_ema --super-htf 4h

# 結果ダウンロード
python3 -m modal run scripts/modal_download.py
```
