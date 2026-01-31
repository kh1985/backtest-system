# Prism

仮想通貨トレード戦略のバックテスト・分析・自動最適化を行うWebダッシュボード。
Streamlit + Plotly で構築。Numba JIT による高速バックテスト対応。

## 機能

### データ
- **データ読み込み**: Binance CSV/ZIP 一括読み込み、TradingView CSV 対応、TF 自動判定
- **データ切り出し**: レンジスライダーで任意期間を切り出し、別データセットとして保持

### 戦略構築・バックテスト
- **戦略ビルダー**: インジケーター・エントリー条件・決済ルールを GUI で設定（YAML 入出力対応）
- **バックテストエンジン**: bar-by-bar シミュレーション（TP/SL/トレーリングストップ/タイムアウト）
- **トレード分析**: 個別トレード詳細チャート・損益分布・勝敗統計・決済タイプ分布

### 自動最適化
- **グリッドサーチ**: 6 種テンプレート × パラメータ全組み合わせ × トレンドレジーム別
- **並列実行**: ProcessPoolExecutor + Numba JIT で高速化
- **バッチ実行**: 複数銘柄 × データソースを一括最適化
- **結果保存**: JSON/CSV 自動保存、フィルタ付き読込ビュー、ファイル削除

### 比較・分析
- **複数銘柄比較**: サマリーマトリクス、レジーム別横断比較、横断分析
- **メタ分析**: 銘柄×レジーム ヒートマップ、テンプレート採択分布、パラメータ収束
- **レジーム別ベスト戦略**: 各レジームの最適戦略カード表示 + YAML 一括エクスポート

## セットアップ

```bash
pip install -e .
```

## 起動

```bash
streamlit run ui/app.py
```

## ページ構成

| ページ | 説明 |
|--------|------|
| **Data** | OHLCV データ読み込み・プレビュー・期間切り出し |
| **Strategy** | 戦略を GUI で作成（YAML 入出力対応） |
| **Backtest** | 戦略を 1 回実行し結果をチャート表示 |
| **Analysis** | トレード単位の深掘り分析 |
| **Optimizer** | グリッドサーチ自動最適化（設定 / 結果 / 読込 / 比較） |

### ワークフロー

```
手動検証:    Data → Strategy → Backtest → Analysis
自動最適化:  Data → Optimizer（設定→実行→結果→YAML export）
銘柄横断:    Data → Optimizer（バッチ実行→比較→メタ分析）
```

## テンプレート

| テンプレート | 概要 |
|-------------|------|
| MA Crossover | SMA fast/slow クロス |
| RSI Reversal | RSI 売られすぎ反発 |
| BB Bounce | ボリンジャーバンド下限タッチ |
| MACD Signal | MACD ラインクロス |
| Volume Spike | 出来高急増 + 陰線反転 |
| Stochastic Reversal | ストキャスティクス K/D クロス |

## トレンドレジーム検出

上位 TF から Uptrend / Downtrend / Range の 3 レジームを判定し、実行 TF に forward-fill。

- **MA Cross**: SMA fast vs SMA slow の方向
- **ADX**: ADX 値でトレンド強度を判定
- **Combined**: MA Cross + ADX の組み合わせ

## 複合スコア

```
score = PF正規化 × w1 + 勝率 × w2 + (1 - DD) × w3 + Sharpe正規化 × w4
```

重みは UI 上でスライダー調整可能（デフォルト: PF 0.3 / WR 0.3 / DD 0.2 / Sharpe 0.2）。

## プロジェクト構成

```
backtest-system/
├── analysis/          # トレンドレジーム検出（MA Cross, ADX, Combined）
├── config/            # 設定（settings.py）
├── data/              # データ読み込み（csv_loader, binance_loader, timeframe）
├── engine/            # バックテストエンジン（backtest.py, numba_loop.py）
├── indicators/        # インジケーター（SMA, EMA, RSI, BB, MACD, ADX, Stoch, VWAP等）
├── metrics/           # パフォーマンス指標計算（calculator.py）
├── optimizer/         # グリッドサーチ・テンプレート・スコアリング・結果
├── results/           # 最適化結果 JSON/CSV 保存先
├── scripts/           # データダウンロードスクリプト
├── strategy/          # 戦略定義・ビルダー・YAML例（examples/）
├── ui/
│   ├── app.py         # メインアプリ（Prism）
│   ├── components/    # 共通コンポーネント（chart, styles, optimizer_charts）
│   └── views/         # 各ページ
│       ├── data_loader.py        # データ読み込み
│       ├── strategy_builder.py   # 戦略ビルダー
│       ├── backtest_runner.py    # バックテスト実行
│       ├── trade_analysis.py     # トレード分析
│       └── optimizer_page.py     # 最適化（設定/結果/読込/比較）
├── inputdata/         # 入力データ置き場
├── sample_data/       # サンプルデータ
└── tests/             # テスト
```

## 技術スタック

- Python 3.10+
- Streamlit（WebUI）
- Plotly（チャート）
- pandas / numpy（データ処理）
- Numba（JIT コンパイル高速化）
- ccxt（取引所 API）
- PyYAML（戦略設定）
