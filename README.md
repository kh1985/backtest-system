# Prism

仮想通貨トレード戦略のバックテスト・分析・自動最適化を行う Web ダッシュボード。
Streamlit + Plotly で構築。Modal クラウド並列実行対応。

## 機能

### データ
- **データ読み込み**: Binance CSV/ZIP 一括読み込み、TradingView CSV 対応、TF 自動判定
- **データ切り出し**: レンジスライダーで任意期間を切り出し
- **Modal データ取得**: Binance API から直接 OHLCV データをクラウド取得

### 戦略構築・バックテスト
- **戦略ビルダー**: インジケーター・エントリー条件・決済ルールを GUI で設定（YAML 入出力対応）
- **バックテストエンジン**: bar-by-bar シミュレーション（TP/SL/トレーリングストップ/タイムアウト）
- **トレード分析**: 個別トレード詳細チャート・損益分布・勝敗統計・決済タイプ分布

### 自動最適化
- **テンプレート**: 21 種単体 + 19 種複合 = 40 テンプレート
- **Exit Profiles**: ATR 固定/トレーリング/VWAP/BB 系 全 69 パターン + コンパクト 3 択モード
- **グリッドサーチ**: テンプレート × パラメータ全組み合わせ × トレンドレジーム別
- **OOS 検証**: train 60% / val 20% / test 20% のアウトオブサンプル検証
- **WFA**: Walk-Forward Analysis（Anchored 5 フォールド）による堅牢性検証
- **Modal 並列実行**: 30 銘柄 × 3 期間を数十秒でクラウド最適化

### 比較・分析
- **複数銘柄比較**: サマリーマトリクス、レジーム別横断比較
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

## Modal クラウド実行

```bash
# データ取得（Binance → Modal Volume）
python3 -m modal run scripts/modal_fetch_data.py

# 最適化実行
python3 -m modal run scripts/modal_optimize.py \
  --period "20230201-20240131,20240201-20250131,20250201-20260130" \
  --exit-profiles atr_compact --super-htf 4h

# 結果ダウンロード
python3 -m modal run scripts/modal_download.py --run-id {RUN_ID}
```

## ローカル WFA 検証

```bash
# デフォルト（4銘柄×3期間、確定戦略）
python3 scripts/local_wfa_test.py

# カスタム
python3 scripts/local_wfa_test.py \
  --symbols AAVEUSDT,BNBUSDT,SOLUSDT \
  --templates rsi_bb_long_f35 \
  --exit-filter atr_tp20_sl20
```

## テンプレート（抜粋）

| テンプレート | 概要 | Long/Short |
|-------------|------|-----------|
| MA Crossover | SMA fast/slow クロス | L+S |
| RSI Reversal | RSI 売られすぎ/買われすぎ反発 | L+S |
| BB Bounce | ボリンジャーバンド下限/上限タッチ | L+S |
| MACD Signal | MACD ラインクロス | L+S |
| Volume Spike | 出来高急増 + 陰線/陽線反転 | L+S |
| Stochastic Reversal | ストキャスティクス K/D クロス | L+S |
| **RSI+BB Reversal** | RSI 閾値 + BB バンドタッチ（複合） | L+S |
| **BB+Volume Reversal** | BB バンド + 出来高急増（複合） | L+S |
| rsi_bb_long_f35 | RSI<35 固定 + BB 下限（実運用確定） | L |

## トレンドレジーム検出

上位 TF から Uptrend / Downtrend / Range の 3 レジームを判定し、実行 TF に forward-fill。

| 方式 | 概要 |
|------|------|
| MA Cross | SMA fast vs SMA slow の方向 |
| ADX | ADX 値でトレンド強度を判定 |
| Combined | MA Cross + ADX の組み合わせ |
| **Dual-TF EMA** | 4h + 1h EMA 合意方式（推奨） |

## 戦略探索の成果

12 ステップの体系的な探索（30 銘柄 × 3 年 × 40 テンプレート）を経て確定した実運用戦略:

| 戦略 | Exit | WFA ROBUST 率 | 平均 PnL | 用途 |
|------|------|--------------|---------|------|
| **rsi_bb_long_f35 / uptrend / tp20** | 固定 | **22% (12/54)** | +9.9% | 安定重視 |
| rsi_bb_long_f35 / uptrend / tp30 | 固定 | 20% (11/54) | +14.1% | 利益重視 |

**主な発見:**
- BB（ボリンジャーバンド）複合が銘柄横断性の鍵
- 探索空間を小さくする = 過学習防止の鍵
- exit 固定が WFA 安定性の鍵（3 択混合 11% → 固定 22%）
- OOS PASS と WFA ROBUST は乖離する → WFA を最終判定に使用

## プロジェクト構成

```
backtest-system/
├── analysis/          # トレンドレジーム検出（MA Cross, ADX, Combined, Dual-TF EMA）
├── config/            # 設定（settings.py）
├── data/              # データ読み込み（csv_loader, binance_loader）
├── engine/            # バックテストエンジン（backtest.py）
├── indicators/        # インジケーター（SMA, EMA, RSI, BB, MACD, ADX, Stoch, VWAP 等）
├── metrics/           # パフォーマンス指標計算（calculator.py）
├── optimizer/         # グリッドサーチ・テンプレート・WFA・exit profiles・スコアリング
├── results/           # 最適化結果 JSON / WFA 結果保存先
├── scripts/           # Modal 実行スクリプト・ローカル WFA・分析スクリプト
├── strategy/          # 戦略定義・ビルダー・YAML 例（examples/）
├── ui/
│   ├── app.py         # メインアプリ（Prism）
│   ├── components/    # 共通コンポーネント（chart, styles, metrics_card）
│   └── views/         # 各ページ（data_loader, strategy_builder, backtest_runner 等）
├── inputdata/         # 入力データ置き場
└── tests/             # テスト
```

## 技術スタック

- Python 3.10+
- Streamlit（WebUI）
- Plotly（チャート）
- pandas / numpy（データ処理）
- Modal（クラウド並列実行）
- ccxt（取引所 API）
- PyYAML（戦略設定）
