# 汎用戦略探索ロードマップ

## ゴール
3銘柄以上のOOSで機能する実運用可能な汎用戦略を見つける

## 固定済みの前提
- レジーム: Dual-TF EMA (4h+1h)
- Exit: atr_compact (SL=ATR×2.0, TP 3択)
- OOS: train60/val20/test20, min_oos_trades=20
- データ: 10銘柄(BTC,ETH,BNB,SOL,XRP,ADA,AVAX,DOGE,DOT,TRX) × 2年(2024-02〜2026-01)

---

## Step 1: ベースライン（全テンプレート混合）— 完了

**Run ID**: 20260206_155925
**条件**: 21テンプレート全部 × atr_compact × 10銘柄

### 結果: OOS PASS 8/30 (27%), 3銘柄共通=0

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL(2件) | FAIL -3.5% | FAIL -1.3% |
| ETH | FAIL(15件) | FAIL -8.0% | FAIL -5.2% |
| BNB | FAIL(17件) | PASS +2.0%(vol_spike_short) | FAIL(19件) |
| SOL | FAIL | PASS +4.9%(ma_crossover) | FAIL |
| XRP | 0件 | PASS +25.5%(rsi_reversal) | PASS +7.1%(vol_spike_short) |
| ADA | PASS +3.1%(bb_bounce) | PASS +6.8%(rsi_reversal) | FAIL |
| AVAX | FAIL | FAIL | FAIL |
| DOGE | FAIL(17件) | FAIL | PASS +0.06%(rsi_rev_short) |
| DOT | FAIL | FAIL | FAIL |
| TRX | FAIL | PASS +5.2%(vol_spike_short) | FAIL |

### 問題点
各銘柄で「一番勝った」テンプレートが選ばれるため、テンプレートがバラバラ。
特定テンプレートが他銘柄でも機能するかは**テストされていない**。

---

## Step 2: テンプレート固定の横断テスト — 実行中

Step 1でOOS PASSしたテンプレートを1つずつ固定し、全10銘柄で強制テスト。
これにより「rsi_reversalは本当に2銘柄だけなのか、テストしてないだけなのか」が判明。

### 候補テンプレート
1. `rsi_reversal` — downtrend 2銘柄(XRP+25.5%, ADA+6.8%)
2. `volume_spike_short` — downtrend 2銘柄(BNB+2.0%, TRX+5.2%) + range 1銘柄(XRP+7.1%)
3. `ma_crossover` — downtrend 1銘柄(SOL+4.9%)
4. `bb_bounce` — uptrend 2銘柄(ADA+3.1%, DOGE+2.96%※17件FAIL)

### 結果記録

#### 2-1. rsi_reversal 固定
- **Run ID**: (未実行)
- **OOS PASS銘柄数**: ?/10

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | | | |
| ETH | | | |
| BNB | | | |
| SOL | | | |
| XRP | | | |
| ADA | | | |
| AVAX | | | |
| DOGE | | | |
| DOT | | | |
| TRX | | | |

#### 2-2. volume_spike_short 固定
- **Run ID**: (未実行)
- **OOS PASS銘柄数**: ?/10

#### 2-3. ma_crossover 固定
- **Run ID**: (未実行)
- **OOS PASS銘柄数**: ?/10

#### 2-4. bb_bounce 固定
- **Run ID**: (未実行)
- **OOS PASS銘柄数**: ?/10

---

## Step 3: 判定

- **3銘柄以上PASS** → エッジ確定。WFAで時間的頑健性を追加検証して保存
- **2銘柄止まり** → 単一テンプレートの限界確定。Step 4へ

---

## Step 4: 複合テンプレート（Step 3で不足の場合のみ）

Step 2で惜しかったテンプレート同士を組み合わせる（例: RSI + Volume）
パラメータは2-3個に抑えて同じ横断テストで検証

---

## 変更ログ
- 2026-02-06: ロードマップ作成、Step 1完了
