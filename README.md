# Backtest System v2

仮想通貨のバックテスト＆戦略自動最適化システム。
Streamlit WebダッシュボードでGUI操作可能。

## 機能

- **データ読み込み**: TradingView CSV / Binance CSV・ZIP / 取引所API (ccxt)
- **戦略ビルダー**: インジケーター・エントリー条件・決済ルールをGUIで設定
- **バックテスト**: bar-by-bar シミュレーション（TP/SL/トレーリングストップ/タイムアウト）
- **トレード分析**: 個別トレードの詳細チャート・損益分布・勝敗統計
- **自動最適化**: 6種テンプレート × パラメータグリッドサーチ × トレンドレジーム別

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
| **Data** | OHLCVデータの読み込み・プレビュー |
| **Strategy** | 戦略をGUIで作成（YAML入出力対応） |
| **Backtest** | 戦略を1回実行し結果をチャート表示 |
| **Analysis** | トレード単位の深掘り分析 |
| **Optimizer** | グリッドサーチで最適な戦略を自動探索 |

### 使い方フロー

```
手動検証:    Data → Strategy → Backtest → Analysis
自動最適化:  Data → Optimizer → (YAML export) → Strategy → Backtest → Analysis
```

## Optimizer

6種の戦略テンプレートをパラメータ全組み合わせで実行し、複合スコアでランキング。

### テンプレート

| テンプレート | 概要 |
|-------------|------|
| MA Crossover | SMA fast/slow クロス |
| RSI Reversal | RSI売られすぎ反発 |
| BB Bounce | ボリンジャーバンド下限タッチ |
| MACD Signal | MACDラインクロス |
| Volume Spike | 出来高急増 + 陰線反転 |
| Stochastic Reversal | ストキャスティクス K/D クロス |

### トレンドレジーム検出

上位TFからUptrend / Downtrend / Rangeの3レジームを判定し、実行TFにforward-fill。

- **MA Cross**: SMA20 vs SMA50 の方向
- **ADX**: ADX値でトレンド強度を判定
- **Combined**: MA Cross + ADX の組み合わせ

### 複合スコア

```
score = PF正規化 × w1 + 勝率 × w2 + (1 - DD) × w3 + Sharpe正規化 × w4
```

重みはUI上でスライダー調整可能（デフォルト: PF 0.3 / WR 0.3 / DD 0.2 / Sharpe 0.2）。

## プロジェクト構成

```
backtest-system/
├── analysis/          # トレンドレジーム検出
├── config/            # 設定（API Key等）
├── data/              # データ読み込み（CSV, Binance, ccxt）
├── engine/            # バックテストエンジン
├── indicators/        # インジケーター（SMA, EMA, RSI, BB, MACD, ADX, etc.）
├── metrics/           # パフォーマンス指標計算
├── optimizer/         # グリッドサーチ・テンプレート・スコアリング
├── scripts/           # データダウンロードスクリプト
├── strategy/          # 戦略定義・ビルダー・YAML例
├── ui/                # Streamlit UI
│   ├── app.py         # メインアプリ
│   ├── components/    # チャート・テーブル等の共通コンポーネント
│   └── pages/         # 各ページ
├── sample_data/       # サンプルデータ（Binance kline）
└── tests/             # テスト
```

## 技術スタック

- Python 3.10+
- Streamlit（WebUI）
- Plotly（チャート）
- pandas / numpy（データ処理）
- ccxt（取引所API）
- PyYAML（戦略設定）
