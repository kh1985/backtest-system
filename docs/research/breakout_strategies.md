# 急変動キャッチ戦略調査レポート

## 調査概要
GogoJungleおよび関連リソースにおける急落・急騰を捉えるブレイクアウト戦略の調査結果。
調査日: 2026-02-11

---

## 1. GogoJungleで公開されている戦略

### 1.1 急騰・急落アラートシステム (zz_BurstNotify)
**特徴:**
- 相場の急騰・急落をリアルタイムでアラート通知
- 勢いのあるポイントを狙い撃ちする設計
- モメンタムトレーディングに特化

**参考:** [相場の急騰／急落をアラート通知](https://www.gogojungle.co.jp/tools/indicators/31170)

### 1.2 Break Scal System（ブレイクスキャルシステム）
**特徴:**
- ブレイクアウトの瞬間を見極めるスキャルピングシステム
- **勝率90%以上を要求**する高精度設計
- ヒットアンドアウェイ型の素早いエントリー・エグジット
- ボリンジャーバンドブレイク + トレンドフォロー

**ロジック概要:**
- ブレイクアウトの判定条件: ボリンジャーバンド突破
- 小さな利益を高頻度で積み重ねる設計
- トレンドフォロー要素を組み込むことでダマシを軽減

**参考:** [Break Scal System](https://www.gogojungle.co.jp/systemtrade/fx/6587)

### 1.3 ボックスブレイクアウト戦略
**特徴:**
- アジア時間・ロンドン時間のボックス相場からのブレイクアウトを狙う
- 東京市場の高値/安値ブレイク、NYボックスブレイクアウト手法
- 時間帯ごとの価格レンジ形成を利用

**参考:** [アジア時間、ロンドン時間のボックス相場からのブレイクアウト](https://www.gogojungle.co.jp/finance/navi/articles/21299)

---

## 2. ブレイクアウト戦略の核心ロジック

### 2.1 ボラティリティブレイクアウトの原理

**基本概念:**
- ボラティリティ（価格変動率）は縮小・拡大のサイクルを繰り返す
- **ボラティリティが長時間縮小するほど、反動で相場が一方向に大きく動く**
- 圧縮されたエネルギーが解放される瞬間を捉える

**検知方法:**
1. **ボラティリティ圧縮の検出**
   - ボリンジャーバンドの収縮（スクイーズ）
   - ATR（Average True Range）の低下
   - レンジ相場の継続期間

2. **ブレイクアウトの確認**
   - サポート/レジスタンスラインの突破
   - ボリンジャーバンドの急拡大（エクスパンション）
   - 出来高の急増（2倍以上が目安）

**参考:**
- [ATRボリンジャーを使ったボラティリティ・ブレイクアウト手法](https://manabu-blog.com/atr-bollinger-volatility-breakout)
- [FXのボラティリティブレイクアウト手法と注意点](https://tips.jp/u/fukugyocharenge/a/BA5kL1Sy)

### 2.2 価格速度（Price Velocity）検知

**モメンタム指標の活用:**
- **RSI (Relative Strength Index)**: モメンタムの強さを数値化
- **MACD (Moving Average Convergence Divergence)**: トレンドの転換点を検出
- **ADX (Average Directional Index)**: トレンドの強さを測定
  - ADX >= 25: 中程度のトレンド
  - ADX >= 40: 強いトレンド（ブレイクアウトが本物の可能性大）

**価格速度の計算:**
- 一定期間の価格変化率（ROC: Rate of Change）
- 移動平均線との乖離率
- ATRの急上昇（ボラティリティ拡大の兆候）

**参考:**
- [ADXインジケーターとは？見方や使い方、DMIの計算方法](https://www.ig.com/jp/trading-strategies/what-is-the-adx-indicator-and-how-do-i-use-it-in-trading-241203)
- [一目遅行線＆ボリンジャーバンド＆DMI](https://www.gaitameonline.com/academy_chart15.jsp)

### 2.3 複合指標による精度向上

**推奨される組み合わせ:**

| 組み合わせ | 役割分担 | エッジの源泉 |
|-----------|---------|-------------|
| **BB + ADX** | BB=ブレイク検知、ADX=トレンド強度確認 | トレンドが強いときのバンドブレイクは本物 |
| **BB + Volume** | BB=ブレイク検知、Volume=需給確認 | 出来高2倍以上で強い市場関心を確認 |
| **ATR BB + MACD** | ATR BB=ボラティリティ正規化、MACD=方向確認 | クロスオーバーでブレイク方向を判定 |
| **BB Squeeze + Moving Average** | Squeeze=圧縮検知、MA=トレンド方向 | 主要MAを基準にブレイク方向を予測 |

**ダマシ回避のポイント:**
- バンドが収束した状態から急激に拡大する瞬間が最も信頼性が高い
- 相場の天底でボリンジャーバンドのエクスパンションが逆方向に発生 = トレンド転換のモメンタムが非常に強い
- 長期間のレンジ相場後のブレイクアウトは大きなトレンドの始まりとなる可能性大

**参考:**
- [相場のボラティリティを解析！ATRボリンジャーの使い方を徹底解説](https://manabu-blog.com/atr-bollinger-trading-strategy)
- [ボリンジャーバンドの最強設定とは？5つのおすすめFX手法](https://a-kufx.com/bollinger-bands-9333.html)

---

## 3. 仮想通貨ブレイクアウト戦略の最新トレンド（2025-2026）

### 3.1 TradingViewのVolatility Momentum Breakout Strategy

**特徴:**
- ボラティリティブレイクアウト + トレンドフィルター + モメンタムフィルターの3層構造
- 重要な価格変動を捉えるための設計
- **Sharpe Ratio 1.0〜1.2**を達成（モメンタム戦略、2021年以前）

**参考:** [Volatility Momentum Breakout Strategy](https://www.tradingview.com/script/dJe0bGvQ-Volatility-Momentum-Breakout-Strategy/)

### 3.2 暗号資産ブレイクアウト検出の最適指標

**推奨指標ランキング:**
1. **Moving Averages (SMA/EMA)**: トレンド方向の確認
2. **RSI**: モメンタムの強度測定
3. **Bollinger Bands**: ボラティリティとブレイクポイント
4. **Volume**: ブレイクアウトの強度を確認（平均の2倍以上が目安）

**組み合わせ手法:**
- **Bollinger Bands + MACD/Moving Averages**:
  - Bollinger Squeeze（バンド収縮）で潜在的な爆発を検知
  - MACDクロスオーバーまたは主要移動平均線を基準にブレイク方向と強度を確認

**出来高確認の重要性:**
- **上昇出来高 + 価格ブレイクアウト = 強いモメンタム**
- 平均の2倍以上の出来高を伴うブレイクアウトは強い市場関心を反映
- 成功確率が大幅に向上

**参考:**
- [Best Indicators for Identifying Crypto Breakouts](https://www.cryptohopper.com/blog/best-indicators-for-identifying-crypto-breakouts-11373)
- [How to Use Volume for High Volatility Breakouts](https://www.luxalgo.com/en/blog/how-to-use-volume-for-high-volatility-breakouts)
- [Volatility Indicators in Crypto Trading 2025](https://zignaly.com/crypto-trading/indicators/volatility-indicators)

### 3.3 システマティックトレーディング戦略

**3つの柱:**
1. **Momentum（モメンタム）**: トレンドフォロー
2. **Mean Reversion（平均回帰）**: 過剰反応からの反転
3. **Volatility Filtering（ボラティリティフィルター）**: 低ボラティリティ期間を除外

**参考:** [Systematic Crypto Trading Strategies](https://medium.com/@briplotnik/systematic-crypto-trading-strategies-momentum-mean-reversion-volatility-filtering-8d7da06d60ed)

---

## 4. Guardian Bot実装のヒント

### 4.1 エントリーロジック（推奨）

```python
# 擬似コード
def detect_breakout_entry():
    # 1. ボラティリティ圧縮の検出
    bb_squeeze = bollinger_bandwidth < threshold  # ATR正規化推奨

    # 2. ブレイクアウトの確認
    price_break = (close > bb_upper) or (close < bb_lower)

    # 3. トレンド強度確認
    strong_trend = adx >= 25  # 40以上ならより強い

    # 4. 出来高確認
    volume_spike = volume > avg_volume * 2.0

    # 5. モメンタム方向確認
    if close > bb_upper:
        momentum_up = (ema_fast > ema_slow) or (macd > signal)
        return momentum_up and strong_trend and volume_spike
    elif close < bb_lower:
        momentum_down = (ema_fast < ema_slow) or (macd < signal)
        return momentum_down and strong_trend and volume_spike

    return False
```

### 4.2 Exit戦略（推奨）

**固定利確/損切り:**
- **UT（Uptrend）**: tp20（2.0% Take Profit）
- **DT（Downtrend）**: tp15（1.5% Take Profit）
- **損切り**: ATR ✕ 2.0 または固定1.5%

**トレーリングストップ（オプション）:**
- ATRベースのトレーリング（ATR ✕ 2.5）
- ボリンジャーバンド反対側タッチで決済

### 4.3 ダマシ回避フィルター

**必須条件:**
1. **ボラティリティ圧縮の事前確認**（スクイーズ期間 >= N本）
2. **ADX >= 25**（トレンド強度フィルター）
3. **出来高 >= 平均 ✕ 2.0**（需給フィルター）
4. **モメンタム指標の合意**（MACD + EMA Crossの同方向確認）

**回避すべきパターン:**
- レンジ相場内の小さなブレイク（ADX < 25）
- 出来高を伴わないブレイク（薄商いのダマシ）
- トレンド末期のブレイク（RSI > 70 or < 30での新規エントリー回避）

### 4.4 Prismシステムへの統合案

**新規テンプレート候補:**
1. **`bb_squeeze_breakout_long`**
   - BB bandwidth < threshold
   - close > bb_upper
   - ADX >= 25
   - volume > avg_volume * 2.0
   - EMA fast cross up

2. **`bb_squeeze_breakout_short`**
   - BB bandwidth < threshold
   - close < bb_lower
   - ADX >= 25
   - volume > avg_volume * 2.0
   - EMA fast cross down

3. **`atr_volatility_breakout_long`**
   - ATR < percentile(20)（低ボラティリティ）
   - close > range_high（レンジ上限突破）
   - ADX >= 25
   - volume spike

**パラメータ探索空間:**
- `bb_period`: [20]（固定推奨）
- `bb_std`: [2.0]（固定推奨）
- `adx_threshold`: [20, 25, 30]
- `volume_multiplier`: [1.5, 2.0, 2.5]
- `ema_fast/slow`: [5/13]（固定推奨）

**Exit設定:**
- UT: tp20 固定
- DT: tp15 固定
- SL: ATR ✕ 2.0

---

## 5. 重要な発見と推奨事項

### 5.1 既存Prism戦略との相違点

| 項目 | 既存戦略（RSI+BB） | ブレイクアウト戦略 |
|------|------------------|------------------|
| **エントリー哲学** | 逆張り（売られすぎ買い） | 順張り（ブレイク順張り） |
| **ボラティリティ** | 高ボラ・低ボラ問わず | **低ボラ圧縮後の拡大を狙う** |
| **トレンド依存** | レジーム検出で分離 | **ADXでトレンド強度必須** |
| **出来高** | 未使用 | **必須フィルター（2倍基準）** |
| **ダマシリスク** | BB lowerタッチのダマシ | BB突破後の反転ダマシ |

### 5.2 実装推奨順位

**優先度1（即実装推奨）:**
- `bb_squeeze_breakout_long/short`: BB収縮→拡大の古典的パターン
- 出来高フィルター追加（既存テンプレートに統合可能）

**優先度2（検証後実装）:**
- `atr_volatility_breakout_long/short`: ATR正規化版
- 時間帯フィルター（アジア/ロンドン/NYボックス）

**優先度3（将来的検討）:**
- 複数時間足確認（4h圧縮 + 1h/15mブレイク）
- 価格速度（ROC）ベースのフィルター

### 5.3 WFA検証の予測

**期待されるROBUST率:**
- **bb_squeeze_breakout**: 25-35%（EMAクロスと同等）
  - 理由: 順張り + ボラティリティフィルター + ADX確認の3重フィルター
  - リスク: トレード数不足（スクイーズ頻度による）

**懸念点:**
- **探索空間が大きい場合は過学習リスク**（ADX閾値 × volume_multiplier × EMA組合せ）
- **トレード数不足**: 圧縮期間の厳しいフィルターでエントリー激減の可能性
- **仮想通貨の高ボラ特性**: FXと異なり常時高ボラ → 圧縮検出が困難かも

**推奨アプローチ:**
1. まず**パラメータ固定版**でOOS検証（ADX=25, volume=2.0, EMA=5/13）
2. OOS PASS率 >= 30% なら WFA実施
3. OOS < 30% ならトレード数を確認 → 不足なら条件緩和

---

## 6. 参考文献・ソース

### GogoJungle関連
- [相場の急騰／急落をアラート通知](https://www.gogojungle.co.jp/tools/indicators/31170)
- [ブレイクアウト型おすすめEAのご紹介](https://www.gogojungle.co.jp/finance/navi/articles/46145)
- [Break Scal System](https://www.gogojungle.co.jp/systemtrade/fx/6587)
- [アジア時間、ロンドン時間のボックス相場からのブレイクアウト](https://www.gogojungle.co.jp/finance/navi/articles/21299)

### ボラティリティブレイクアウト手法
- [ATRボリンジャーを使ったボラティリティ・ブレイクアウト手法](https://manabu-blog.com/atr-bollinger-volatility-breakout)
- [FXのボラティリティブレイクアウト手法と注意点](https://tips.jp/u/fukugyocharenge/a/BA5kL1Sy)
- [相場のボラティリティを解析！ATRボリンジャーの使い方を徹底解説](https://manabu-blog.com/atr-bollinger-trading-strategy)

### ブレイクアウト戦略基礎
- [FXのブレイクアウト手法とは？メリットやデメリット、有効な手法を解説｜IG証券](https://www.ig.com/jp/trading-strategies/guide-to-trading-breakouts-in-forex-250121)
- [FX初心者がブレイクアウトの「ダマシ」を回避するコツ](https://www.gaitame.com/media/entry/2020/06/02/151617)
- [Top 5 Breakout Trading Strategies That Actually Work](https://www.ebc.com/forex/top-breakout-trading-strategies-that-actually-work)

### 暗号資産ブレイクアウト戦略（2025-2026）
- [Volatility Momentum Breakout Strategy by cryptechcapital — TradingView](https://www.tradingview.com/script/dJe0bGvQ-Volatility-Momentum-Breakout-Strategy/)
- [Best Indicators for Identifying Crypto Breakouts](https://www.cryptohopper.com/blog/best-indicators-for-identifying-crypto-breakouts-11373)
- [How to Use Volume for High Volatility Breakouts](https://www.luxalgo.com/blog/how-to-use-volume-for-high-volatility-breakouts)
- [Volatility Indicators in Crypto Trading - Master Market Swings Like a Pro in 2025](https://zignaly.com/crypto-trading/indicators/volatility-indicators)
- [How to Trade Breakouts in Crypto: Strategies, Tips, and Risk Management](https://flipster.io/blog/how-to-trade-breakouts-in-crypto-strategies-tips-and-risk-management)

### テクニカル指標解説
- [ADXインジケーターとは？見方や使い方、DMIの計算方法](https://www.ig.com/jp/trading-strategies/what-is-the-adx-indicator-and-how-do-i-use-it-in-trading-241203)
- [一目遅行線＆ボリンジャーバンド＆DMI](https://www.gaitameonline.com/academy_chart15.jsp)
- [ボリンジャーバンドの最強設定とは？5つのおすすめFX手法](https://a-kufx.com/bollinger-bands-9333.html)

### システマティック戦略
- [Systematic Crypto Trading Strategies: Momentum, Mean Reversion & Volatility Filtering](https://medium.com/@briplotnik/systematic-crypto-trading-strategies-momentum-mean-reversion-volatility-filtering-8d7da06d60ed)

---

## 付録: 用語集

| 用語 | 説明 |
|------|------|
| **Bollinger Squeeze** | ボリンジャーバンドが収縮し、価格変動が縮小している状態。エネルギー圧縮を示唆 |
| **Expansion（エクスパンション）** | バンドが急激に拡大する状態。ボラティリティ拡大の証拠 |
| **ATR (Average True Range)** | 平均真の値幅。ボラティリティを数値化した指標 |
| **ADX (Average Directional Index)** | 平均方向性指数。トレンドの強さを測定（25以上=トレンド、40以上=強トレンド） |
| **Volume Spike** | 出来高急増。通常の2倍以上が目安 |
| **False Breakout（ダマシ）** | ライン突破後すぐに元の範囲に戻る現象 |
| **Price Velocity** | 価格速度。一定期間の価格変化率（ROC等で測定） |
| **Sharpe Ratio** | シャープレシオ。リスク調整後リターンの指標（1.0以上が良好） |

---

**調査完了日**: 2026-02-11
**担当**: Sonnet Agent (Task #2)
**次のアクション**: チームリーダーへの報告 + Prism既存戦略との統合検討
