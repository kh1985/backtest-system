# /check-results - 最適化結果サマリー

`results/` フォルダの最適化結果ファイルを一覧・サマリー表示する。

## 手順

1. `results/*.json` を列挙する
2. 各JSONファイルを開き、以下の情報を抽出:
   - `symbol`: 銘柄
   - `execution_tf`: 実行タイムフレーム
   - `htf`: 上位タイムフレーム
   - `data_source`: original / trimmed
   - `data_period`: 切り出し期間（trimmedの場合）
   - `results` の件数
   - ベストスコア（results内のscoreの最大値）
   - `timestamp`: 保存日時
3. 結果を以下の形式で表示:

```
| 銘柄     | TF      | データ      | 件数  | ベスト | 保存日時         |
|----------|---------|-------------|-------|--------|------------------|
| BTCUSDT  | 1m→15m  | 📦 全期間   | 1,080 | 0.672  | 01/31 15:38      |
| ETHUSDT  | 1m→15m  | ✂️ 03/19~10/09 | 1,080 | 0.583  | 01/31 15:54  |
```

4. 末尾にサマリー: 合計ファイル数、銘柄数、TF別の内訳

## 引数

- 引数なし: 全ファイルの一覧
- 銘柄名指定（例: `/check-results BTCUSDT`）: その銘柄のファイルだけ表示し、レジーム別ベストスコアも追加表示

## 注意

- 作業ディレクトリは `/Users/kenjihachiya/Desktop/work/development/backtest-system`
- ファイルが0件の場合は「結果ファイルなし」と報告
- 出力は日本語で簡潔に
