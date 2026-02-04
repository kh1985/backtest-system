# Backtest System ガイド

データローディングから自動最適化パイプラインまでの全体像。

---

## 1. システム全体像

```mermaid
flowchart TB
    subgraph INPUT["データ入力"]
        CSV["Binance CSV/ZIP\n(銘柄 x TF x 期間)"]
    end

    subgraph LOAD["データロード"]
        BL["BinanceCSVLoader"]
        OHLCV["OHLCVData\n(DataFrame + メタ情報)"]
    end

    subgraph ANALYSIS["トレンド分析"]
        TD["TrendDetector"]
        REG["レジーム判定\nuptrend / downtrend / range"]
    end

    subgraph INDICATOR["インジケーター計算"]
        IND["11種のインジケーター\nSMA, EMA, RSI, MACD,\nBollinger, Stochastic,\nATR, VWAP, ADX ..."]
    end

    subgraph OPTIMIZE["自動最適化"]
        TPL["14テンプレート\n(エントリー戦略)"]
        EP["10 Exit Profiles\n(exit戦略)"]
        GRID["GridSearchOptimizer\n(Numba高速バックテスト)"]
        OOS["OOS 3分割検証\nTrain 60% / Val 20% / Test 20%"]
    end

    subgraph OUTPUT["結果出力"]
        RANK["自動ランキング\n(スコアリング)"]
        REPORT["Markdownレポート"]
        JSON["JSON結果ファイル"]
    end

    CSV --> BL --> OHLCV
    OHLCV --> TD --> REG
    OHLCV --> IND
    REG --> GRID
    IND --> GRID
    TPL --> GRID
    EP --> GRID
    GRID --> OOS --> RANK
    RANK --> REPORT
    RANK --> JSON
```

---

## 2. データパイプライン

### 2.1 データの準備

Binance公式データ（data.binance.vision）からダウンロードしたCSV/ZIPを使用。

```
inputdata/
├── BTCUSDT-15m-20240201-20250131-merged.csv
├── BTCUSDT-15m-20250201-20260130-merged.csv
├── BTCUSDT-1h-20240201-20250131-merged.csv
├── BTCUSDT-1m-20240201-20250131-merged.csv
├── BTCUSDT-4h-20240201-20250131-merged.csv
├── ETHUSDT-15m-20240201-20250131-merged.csv
└── ...
```

**ファイル命名規則**: `{銘柄}-{タイムフレーム}-{開始日}-{終了日}-merged.csv`

**対応タイムフレーム**: 1m, 5m, 15m, 30m, 1h, 4h, 1d

### 2.2 ダウンロードスクリプト

```bash
# 複数銘柄を一括ダウンロード（1年分）
python scripts/download_multi_1y.py

# 特定期間のデータを補完
python scripts/backfill_data_period.py
```

### 2.3 データロードの仕組み

```mermaid
flowchart LR
    subgraph loader["BinanceCSVLoader"]
        direction TB
        A["CSVファイル読み込み"] --> B["カラム正規化\n(datetime, open, high,\nlow, close, volume)"]
        B --> C["タイムフレーム自動検出\n(ファイル名から)"]
        C --> D["OHLCVData生成"]
    end

    subgraph ohlcv["OHLCVData"]
        direction TB
        DF["DataFrame\n(OHLCV + datetime index)"]
        META["メタ情報\n- symbol\n- timeframe\n- source"]
    end

    loader --> ohlcv
```

**OHLCVData** はシステム内の統一データ形式。全モジュールがこの形式を前提に動作する。

---

## 3. トレンドレジーム検出

上位足（HTF）の移動平均クロスで相場環境を3分類し、レジームごとに別戦略を適用。

```mermaid
flowchart TB
    subgraph htf["上位足データ (例: 1h)"]
        H["1hチャート"]
    end

    subgraph detect["TrendDetector"]
        MA["MA Cross\nSMA(20) vs SMA(50)"]
        ADX_D["ADX\nADX値 + DI+/DI-"]
        COMB["Combined\nMA + ADX 統合"]
    end

    subgraph regime["レジーム分類"]
        UP["UPTREND\nSMA20 > SMA50"]
        DOWN["DOWNTREND\nSMA20 < SMA50"]
        RANGE["RANGE\n差が小さい / ADX低い"]
    end

    subgraph apply["レジーム適用"]
        EXEC["実行足 (例: 15m) に\nレジームラベルを付与\n(前方参照バイアス防止)"]
    end

    htf --> detect
    MA --> regime
    ADX_D --> regime
    COMB --> regime
    regime --> apply
```

| 検出方法 | ロジック | 用途 |
|---------|---------|------|
| `ma_cross` | SMA(fast) vs SMA(slow) の位置関係 | デフォルト。シンプルで安定 |
| `adx` | ADX > 25 → トレンド、< 20 → レンジ | ボラティリティ感応 |
| `combined` | MA + ADX の統合判定 | より精密だがノイズも多い |

---

## 4. エントリーテンプレート

14種の組み込みテンプレートがあり、各テンプレートがパラメータ範囲を持つ。

### 4.1 テンプレート一覧

```mermaid
flowchart LR
    subgraph long["ロング戦略 (6種)"]
        L1["ma_crossover\nSMA高速/低速クロス"]
        L2["rsi_reversal\nRSI売られすぎ反発"]
        L3["bb_bounce\nボリバン下限+RSI"]
        L4["macd_signal\nMACDシグナルクロス"]
        L5["volume_spike\n出来高急増+陽線"]
        L6["stochastic_reversal\nストキャスK/D反転"]
    end

    subgraph short["ショート戦略 (6種)"]
        S1["ma_crossover_short"]
        S2["rsi_reversal_short"]
        S3["bb_bounce_short"]
        S4["macd_signal_short"]
        S5["volume_spike_short"]
        S6["stochastic_reversal_short"]
    end

    subgraph trend["トレンド追随 (2種)"]
        T1["trend_pullback_long\n上昇トレンド押し目"]
        T2["trend_pullback_short\n下降トレンド戻り"]
    end
```

### 4.2 テンプレートの構造

各テンプレートは以下の要素で構成される:

```yaml
name: "ma_crossover"
side: long

# 使用するインジケーター
indicators:
  - type: sma
    period: "{sma_fast}"    # ← パラメータ（グリッドサーチ対象）
  - type: sma
    period: "{sma_slow}"

# エントリー条件
entry_conditions:
  - type: crossover
    fast: "sma_{sma_fast}"
    slow: "sma_{sma_slow}"
    direction: above

entry_logic: and

# 決済条件
exit:
  take_profit_pct: 2.0
  stop_loss_pct: 1.0
```

### 4.3 パラメータ範囲

テンプレートごとに探索するパラメータ範囲が定義されている。

```python
# 例: ma_crossover
param_ranges = [
    ParameterRange("sma_fast", min=10, max=20, step=5, type="int"),
    ParameterRange("sma_slow", min=40, max=60, step=10, type="int"),
]
# → (10,40), (10,50), (10,60), (15,40), ... = 9通り
```

全テンプレートの直積でエントリー条件のバリエーションが生成される。

---

## 5. Exit Profiles（決済戦略）

エントリーとは独立した軸として、10種類のexit戦略を定義。

```mermaid
flowchart TB
    subgraph fixed["固定% モード (3種)"]
        F1["fixed_2_1\nTP=2% / SL=1%"]
        F2["fixed_3_1.5\nTP=3% / SL=1.5%"]
        F3["fixed_1.5_0.5\nTP=1.5% / SL=0.5%"]
    end

    subgraph atr["ATR モード (3種)"]
        A1["atr_3_1.5\nTP=ATR×3 / SL=ATR×1.5"]
        A2["atr_4_2\nTP=ATR×4 / SL=ATR×2"]
        A3["atr_2_1\nTP=ATR×2 / SL=ATR×1"]
    end

    subgraph nosl["SLなし モード (2種)"]
        N1["no_sl_trail2\nトレーリング2%\nタイムアウト50本"]
        N2["no_sl_trail3\nトレーリング3%\nタイムアウト80本"]
    end

    subgraph hybrid["ハイブリッド (2種)"]
        H1["trail2_sl1\nトレーリング2% + SL1%"]
        H2["atr_trail\nATR SL + トレーリング2%"]
    end
```

### 固定% vs ATR の違い

```mermaid
flowchart LR
    subgraph fixed_mode["固定%モード"]
        direction TB
        FE["エントリー: $100"]
        FTP["TP: $102 (100×1.02)"]
        FSL["SL: $99 (100×0.99)"]
        FE --> FTP
        FE --> FSL
    end

    subgraph atr_mode["ATRモード (ATR=$5)"]
        direction TB
        AE["エントリー: $100"]
        ATP["TP: $115 (100+5×3)"]
        ASL["SL: $92.5 (100-5×1.5)"]
        AE --> ATP
        AE --> ASL
    end
```

| モード | メリット | デメリット |
|--------|---------|-----------|
| 固定% | シンプル、予測可能 | ボラに適応しない。SLハントされやすい |
| ATR | ボラティリティに自動適応。TF間で一貫 | ATR計算が必要。短期ではノイズに弱い |
| SLなし | SLハント回避。利益を最大限伸ばせる | ドローダウンが大きくなりうる |
| ハイブリッド | バランス型 | パラメータが増える |

### テンプレート × Exit Profile の直積

```mermaid
flowchart LR
    TPL["14テンプレート\n(各テンプレート内で\nパラメータ直積)"]
    EP["10 Exit Profiles"]
    COMBO["合計 ~4,680 Config\n(テンプレートあたり\n~468パラメータ組み合わせ)"]

    TPL -->|"×"| EP -->|"="| COMBO
```

---

## 6. グリッドサーチ最適化

### 6.1 処理フロー

```mermaid
flowchart TB
    subgraph prep["前処理"]
        DATA["OHLCVデータ"]
        IND["インジケーター\n事前計算\n(55設定を一括計算)"]
        SIG["エントリーシグナル\nベクトル化\n(pandas → bool配列)"]
    end

    subgraph grid["グリッドサーチ"]
        CONFIGS["~4,680 Config生成\n(テンプレート×exit×パラメータ)"]
        NUMBA["Numba JITバックテスト\n(C言語速度の\nポジション管理ループ)"]
        SCORE["スコアリング\n(複合メトリクス)"]
    end

    subgraph result["結果"]
        RANK["スコア順ランキング"]
        TOP["レジーム別 Top戦略"]
    end

    DATA --> IND --> SIG
    SIG --> NUMBA
    CONFIGS --> NUMBA
    NUMBA --> SCORE --> RANK --> TOP
```

### 6.2 スコアリング

4つのメトリクスを加重平均してスコアを算出:

```
composite_score = PF × 0.3 + WinRate × 0.3 + (1 - MaxDD) × 0.2 + Sharpe × 0.2
```

| メトリクス | 重み | 意味 |
|-----------|------|------|
| Profit Factor | 30% | 総利益 / 総損失 |
| Win Rate | 30% | 勝率 |
| Max Drawdown | 20% | 最大ドローダウン（低いほど良い） |
| Sharpe Ratio | 20% | リスク調整リターン |

### 6.3 Numbaバックテストループ

ポジション管理ループをNumba JITでコンパイルし、C言語並みの速度で実行。

```mermaid
flowchart TB
    subgraph loop["_backtest_loop (Numba JIT)"]
        direction TB
        START["各バーをループ"]
        CHECK_POS{"ポジション\nあり?"}
        CHECK_EXIT{"決済条件\n成立?"}
        EXIT_ACT["決済実行\n(TP/SL/トレーリング/タイムアウト)"]
        CHECK_ENTRY{"エントリー\nシグナル?"}
        ENTRY_ACT["エントリー実行\n+ TP/SL価格計算"]
        NEXT["次のバーへ"]

        START --> CHECK_POS
        CHECK_POS -->|Yes| CHECK_EXIT
        CHECK_POS -->|No| CHECK_ENTRY
        CHECK_EXIT -->|Yes| EXIT_ACT --> CHECK_ENTRY
        CHECK_EXIT -->|No| NEXT
        CHECK_ENTRY -->|Yes| ENTRY_ACT --> NEXT
        CHECK_ENTRY -->|No| NEXT
    end
```

**TP/SL価格計算** (`_compute_tp_sl`):

```
■ 固定%モード (ロング):
  TP = entry_price × (1 + tp_pct / 100)
  SL = entry_price × (1 - sl_pct / 100)

■ ATRモード (ロング):
  TP = entry_price + ATR × tp_mult
  SL = entry_price - ATR × sl_mult

■ SLなし:
  SL = -1.0 (ロング) / 1e18 (ショート) → 到達不可能
```

---

## 7. OOS（Out-of-Sample）検証

過学習を防ぐため、データを3分割して段階的に検証。

```mermaid
flowchart LR
    subgraph data["時系列データ"]
        direction LR
        TRAIN["Train 60%\n(グリッドサーチ)"]
        VAL["Val 20%\n(Top-10 再評価)"]
        TEST["Test 20%\n(最終検証)"]
    end

    subgraph flow["検証フロー"]
        direction TB
        S1["1. Train区間で\n全Config探索\n→ スコア順ランキング"]
        S2["2. Val区間で\nTop-10を再評価\n→ ベスト1を選出"]
        S3["3. Test区間で\nベスト戦略を最終検証\n→ OOS結果"]
    end

    TRAIN -.-> S1
    VAL -.-> S2
    TEST -.-> S3
    S1 --> S2 --> S3
```

| フェーズ | 目的 | やること |
|---------|------|---------|
| Train (60%) | 戦略探索 | 全4,680コンボをグリッドサーチ |
| Val (20%) | 選別 | Train上位10戦略をVal区間で再テスト |
| Test (20%) | 最終判定 | Val最良の1戦略のみをテスト（1回限り） |

**OOS通過基準**: Test区間でPnL > 0 かつ十分なトレード数

---

## 8. バッチ自動最適化パイプライン

全条件を一括実行するスクリプト。

### 8.1 実行コマンド

```bash
# フル実行（全TF × 全銘柄 × 全期間 × 全exit profiles）
python scripts/batch_optimize.py

# TF指定
python scripts/batch_optimize.py --tf-combos 15m:1h,15m:4h,1h:4h

# exit profilesを限定
python scripts/batch_optimize.py --exit-profiles fixed

# 小規模テスト
python scripts/batch_optimize.py \
  --symbols BTCUSDT,ETHUSDT \
  --periods 20250201-20260130 \
  --tf-combos 15m:1h

# OOSなし（高速）
python scripts/batch_optimize.py --no-oos
```

### 8.2 CLIオプション一覧

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--symbols` | 全銘柄 | 対象銘柄（カンマ区切り） |
| `--periods` | 2024 + 2025 | 対象期間（カンマ区切り） |
| `--tf-combos` | all (6種) | TF組み合わせ |
| `--exit-profiles` | all (10種) | Exit profiles |
| `--workers` | 4 | 並列ワーカー数 |
| `--oos` | ON | OOS 3分割検証を有効化 |
| `--no-oos` | - | OOS検証なし |
| `--force` | - | 既存結果を上書き |

### 8.3 TF組み合わせ

実行足（エントリー判定）と上位足（トレンド判定）のペア。

| 実行足 | 上位足 | 用途 |
|--------|-------|------|
| 1m | 15m | 超短期スキャルピング |
| 1m | 1h | 短期スキャルピング |
| **15m** | **1h** | **中期（標準）** |
| 15m | 4h | 中期スイング |
| 1h | 4h | スイング |
| 1h | 1d | 長期スイング |

### 8.4 パイプライン全体フロー

```mermaid
flowchart TB
    subgraph phase1["Phase 1: データスキャン"]
        SCAN["inputdata/ をスキャン\n→ 利用可能な\n銘柄×TF×期間を特定"]
    end

    subgraph phase2["Phase 2: 全銘柄最適化"]
        LOOP["FOR each TF組み合わせ\n  FOR each 期間\n    FOR each 銘柄"]
        LOAD["データロード\n(実行足 + 上位足)"]
        TREND["トレンドラベル付与\n(MA Cross 20/50)"]
        GEN["Config生成\n(14テンプレート × 10 exit)"]
        OOS_RUN["OOS 3分割検証実行"]
        SAVE["結果JSON保存"]

        LOOP --> LOAD --> TREND --> GEN --> OOS_RUN --> SAVE
    end

    subgraph phase3["Phase 3: 自動ランキング"]
        COLLECT["全結果を集約"]
        SCORE_R["ランキングスコア算出\nOOS通過率 × 0.4\n+ 平均PnL × 0.3\n+ 期間一貫性 × 0.2\n+ 銘柄カバレッジ × 0.1"]
        FILTER["フィルタリング\nOOS通過率 >= 50%\n+ 3銘柄以上"]
    end

    subgraph phase4["Phase 4: レポート生成"]
        MD["Markdownレポート\n- 推奨戦略ランキング\n- Exit戦略比較\n- TF別比較\n- レジーム別詳細"]
    end

    phase1 --> phase2 --> phase3 --> phase4
```

### 8.5 出力ディレクトリ構造

```
results/batch/{timestamp}/
├── config.json                  # 実行設定
├── optimization/                # 銘柄別OOS結果
│   ├── BTCUSDT_20240201-20250131_15m_1h.json
│   ├── BTCUSDT_20250201-20260130_15m_1h.json
│   ├── ETHUSDT_20240201-20250131_15m_1h.json
│   └── ...
├── ranking.json                 # 自動ランキング結果
└── report.md                    # 最終Markdownレポート
```

### 8.6 自動ランキングスコア

```
score = OOS通過率 × 0.4
      + normalize(平均OOS PnL) × 0.3
      + 期間一貫性 × 0.2
      + 銘柄カバレッジ × 0.1
```

| 指標 | 重み | 意味 |
|------|------|------|
| OOS通過率 | 40% | Test区間で利益が出た割合 |
| 平均OOS PnL | 30% | OOS通過時の平均リターン |
| 期間一貫性 | 20% | 2024年と2025年の両方で通過しているか |
| 銘柄カバレッジ | 10% | 何銘柄で有効だったか |

---

## 9. レジーム × 戦略 マトリクス

各レジームに最適な戦略を自動選択する仕組み。

```mermaid
flowchart TB
    subgraph market["相場環境"]
        UP["UPTREND\n(上昇トレンド)"]
        DOWN["DOWNTREND\n(下降トレンド)"]
        RNG["RANGE\n(レンジ相場)"]
    end

    subgraph strategy["レジーム別最適戦略"]
        UP_S["ロング戦略\n(ma_crossover,\nrsi_reversal, ...)"]
        DOWN_S["ショート戦略\n(ma_crossover_short,\nmacd_signal_short, ...)"]
        RNG_S["レンジ戦略\n(bb_bounce,\nstochastic_reversal, ...)"]
    end

    subgraph switch["レジームスイッチング"]
        RS["_backtest_loop_regime_switching\n現在のレジームに応じて\n使う戦略を自動切替"]
    end

    UP --> UP_S --> RS
    DOWN --> DOWN_S --> RS
    RNG --> RNG_S --> RS
```

---

## 10. ディレクトリ構成

```
backtest-system/
├── analysis/           # トレンドレジーム検出
│   └── trend.py        #   TrendDetector (MA Cross / ADX / Combined)
├── config/             # グローバル設定
│   └── settings.py
├── data/               # データローディング
│   ├── base.py         #   OHLCVData
│   ├── binance_loader.py  # BinanceCSVLoader
│   └── csv_loader.py   #   汎用CSVLoader
├── engine/             # バックテストエンジン
│   └── numba_loop.py   #   Numba JIT ループ
├── indicators/         # テクニカルインジケーター (11種)
│   └── registry.py     #   INDICATOR_REGISTRY
├── metrics/            # メトリクス計算
│   └── calculator.py   #   BacktestMetrics
├── optimizer/          # 最適化エンジン
│   ├── grid.py         #   GridSearchOptimizer
│   ├── templates.py    #   14 BUILTIN_TEMPLATES
│   ├── exit_profiles.py #  10 Exit Profiles
│   ├── validation.py   #   OOS 3分割検証
│   ├── scoring.py      #   複合スコアリング
│   ├── results.py      #   OptimizationResultSet
│   └── regime_switching.py # レジームスイッチング
├── strategy/           # 戦略定義
│   ├── base.py         #   ExitRule, EntryCondition
│   └── builder.py      #   ConfigStrategy
├── scripts/            # バッチスクリプト
│   └── batch_optimize.py  # 全自動最適化パイプライン
├── inputdata/          # 入力データ (CSV)
├── results/            # 最適化結果 (JSON/Markdown)
└── tests/              # テスト (117件)
```
