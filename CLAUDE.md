# Project: Prism (backtest-system)

## Goal

仮想通貨トレード戦略のバックテスト・分析・自動最適化を行うWebダッシュボードシステム。

- OHLCVデータの読み込み（Binance CSV/ZIP, TradingView CSV）
- GUI上で戦略を組み立てて1回バックテスト実行・分析
- 6種テンプレート × パラメータグリッドサーチ × トレンドレジーム別の自動最適化
- 最適化結果のJSON保存・読み込み・YAML戦略エクスポート

## Current state

### 完成済み機能
- **データローダー**: Binance CSV/ZIP一括読み込み、TradingView CSV対応、TF自動判定
- **戦略ビルダー**: インジケーター・エントリー条件・決済ルールをGUIで設定（YAML入出力対応）
- **バックテストエンジン**: bar-by-bar シミュレーション（TP/SL/トレーリングストップ/タイムアウト）
- **トレード分析**: 個別トレード詳細チャート・損益分布・勝敗統計・決済タイプ分布
- **自動最適化**: 6種テンプレート × グリッドサーチ × レジーム別。複合スコアでランキング
- **最適化結果JSON読み込み**: `results/` フォルダのJSON → 結果ビュー表示（`OptimizationResultSet.from_json()`）
- **レジーム別ベスト戦略サマリー**: 各レジームの最適戦略カード表示 + YAML一括エクスポート
- **UI**: Streamlit WebUI、ダークテーマ、カスタムCSS、ワークフローガイド

### テンプレート一覧
| テンプレート | 概要 |
|-------------|------|
| MA Crossover | SMA fast/slow クロス |
| RSI Reversal | RSI売られすぎ反発 |
| BB Bounce | ボリンジャーバンド下限タッチ |
| MACD Signal | MACDラインクロス |
| Volume Spike | 出来高急増 + 陰線反転 |
| Stochastic Reversal | ストキャスティクス K/D クロス |

### UI名称
- システム名: **Prism**（旧: Backtest System）
- ページタイトル: `Prism`
- サイドバー: `🔷 Prism` + `戦略バックテスト＆最適化ツール`

## Architecture

```
backtest-system/
├── analysis/          # トレンドレジーム検出（MA Cross, ADX, Combined）
├── config/            # 設定（settings.py: STREAMLIT_PAGE_TITLE等）
├── data/              # データ読み込み（csv_loader, binance_loader, base）
├── engine/            # バックテストエンジン（backtest.py）
├── indicators/        # インジケーター（SMA, EMA, RSI, BB, MACD, ADX, Stoch, VWAP等）
├── metrics/           # パフォーマンス指標計算（calculator.py: BacktestMetrics）
├── optimizer/         # グリッドサーチ・テンプレート・スコアリング・結果（results.py）
├── results/           # 最適化結果JSON保存先
├── scripts/           # データダウンロードスクリプト
├── strategy/          # 戦略定義・ビルダー・YAML例（examples/）
├── ui/
│   ├── app.py         # メインアプリ（Prism）
│   ├── components/    # 共通コンポーネント（chart, styles, metrics_card, trade_table）
│   └── views/         # 各ページ
│       ├── data_loader.py        # データ読み込みページ
│       ├── strategy_builder.py   # 戦略ビルダーページ
│       ├── backtest_runner.py    # バックテスト実行ページ
│       ├── trade_analysis.py     # トレード分析ページ
│       └── optimizer_page.py     # 最適化ページ（設定/結果/読込の3ビュー）
├── inputdata/         # 入力データ置き場
├── sample_data/       # サンプルデータ
└── tests/             # テスト
```

### Key data structures
- `OptimizationEntry`: 1パラメータ組み合わせの結果（template_name, params, trend_regime, metrics, composite_score）
- `OptimizationResultSet`: 最適化結果セット（entries, symbol, execution_tf, htf）。`from_json()` でJSON復元可能
- `BacktestMetrics`: バックテスト指標（trades, win_rate, profit_factor, max_dd, sharpe等）
- `BacktestResult`: バックテスト結果（trades, df, portfolio）

## Tech stack

- Python 3.10+
- Streamlit（WebUI）
- Plotly（チャート）
- pandas / numpy（データ処理）
- ccxt（取引所API）
- PyYAML（戦略設定）

## Coding style

- Python 3.10+
- 関数は小さくテストしやすい単位で
- ログ出力と例外処理を明確に
- ファイル名・関数名・クラス名・変数名は英語
- UI表示テキスト・コメント・説明は日本語

### パフォーマンス最優先ルール（必須）

数値データ・OHLCV処理では**常に最速の実装**を選ぶこと。1分足データ（数十万行）を扱うため、遅いコードは致命的。

**禁止パターン:**
- `for row in df.iterrows()` / `for row in df.itertuples()` でのループ処理
- Python for文による行単位のDataFrame操作
- リスト内包表記で1行ずつ計算してからDataFrameに変換

**必須パターン:**
- pandas のベクトル演算（`df["col"] = df["a"] + df["b"]`）
- numpy の配列演算（`np.where()`, `np.maximum()`, `np.cumsum()` 等）
- pandas の `.shift()`, `.rolling()`, `.expanding()`, `.cummax()`, `.clip()` 等
- boolean indexing（`df[df["rsi"] < 30]`）

**判断基準:** 「Pythonのforループで書けるが、numpy/pandasのベクトル演算でも書ける」場合は、**例外なくベクトル演算を使う**。

## Workflow rules for Claude Code

- 複雑な変更は必ず最初にPLAN提示
- 変更は論理的な単位で小さく進める
- ファイル削除や構造変更は事前確認必須
- **ユーザーの明示的な指示なく勝手に作業を進めない**

## Language rules

- 説明・計画・コメントは全て日本語
- ファイル名・ディレクトリ名・関数名・クラス名・コード識別子は英語
- 明示的に求められない限り英語に切り替えない

## Discussion notes

### 下落トレンド銘柄横断仮説（未着手）
- ユーザーの仮説: 下落トレンドでは銘柄を問わず特定の戦略（ショート系テンプレート）が機能するのではないか
- 検証アプローチ案: 複数銘柄（TRXUSDT, XRPUSDT等）のdowntrendレジーム結果を横断比較
- ステータス: 議論のみ。実装指示は未出
