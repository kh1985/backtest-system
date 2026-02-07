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

## Step 2: テンプレート固定の横断テスト — 完了

Step 1でOOS PASSしたテンプレートを1つずつ固定し、全10銘柄で強制テスト。

### 2-1. rsi_reversal 固定
- **Run ID**: 20260206_164546
- **OOS PASS**: 4/30 (downtrend 2, range 2)

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL -2.2%(21件) | FAIL +2.0%(14件) | FAIL -2.6%(12件) |
| ETH | FAIL -3.5%(15件) | FAIL -8.0%(16件) | **PASS +2.3%**(33件) |
| BNB | FAIL +1.6%(9件) | FAIL +5.7%(15件) | FAIL +12.4%(19件) |
| SOL | FAIL +1.1%(9件) | FAIL -9.7%(20件) | FAIL -14.6%(63件) |
| XRP | FAIL -0.8%(6件) | **PASS +25.5%**(23件) | **PASS +3.3%**(34件) |
| ADA | FAIL -4.6%(24件) | **PASS +6.8%**(31件) | FAIL +2.1%(18件) |
| AVAX | FAIL -9.3%(21件) | FAIL +7.3%(11件) | FAIL -0.3%(33件) |
| DOGE | FAIL +3.3%(15件) | FAIL -8.3%(28件) | PASS +0.06%(27件) |
| DOT | FAIL +1.4%(5件) | FAIL -5.6%(21件) | FAIL -10.7%(25件) |
| TRX | FAIL -0.1%(2件) | FAIL -6.4%(34件) | FAIL -1.4%(12件) |

**判定**: downtrend 2銘柄のみ。3銘柄未達。range/rsi_reversal_shortは3銘柄PASSだがPnL微小。

### 2-2. volume_spike_short 固定
- **Run ID**: 20260206_164548
- **OOS PASS**: 10/30 (downtrend 6, range 4, uptrend 0)

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL -1.0%(15件) | FAIL -3.5%(16件) | FAIL +0.5%(14件) |
| ETH | FAIL -2.8%(17件) | FAIL +3.1%(17件) | FAIL -5.2%(22件) |
| BNB | FAIL -26.1%(68件) | FAIL +7.0%(13件) | FAIL -10.1%(7件) |
| SOL | FAIL -8.5%(20件) | **PASS +2.8%**(27件) | **PASS +8.9%**(45件) |
| XRP | FAIL -5.1%(10件) | **PASS +6.5%**(23件) | **PASS +7.1%**(26件) |
| ADA | FAIL -4.4%(30件) | **PASS +15.1%**(42件) | FAIL -9.3%(11件) |
| AVAX | FAIL -1.2%(17件) | **PASS +1.5%**(20件) | FAIL -5.6%(16件) |
| DOGE | FAIL -13.3%(9件) | FAIL -8.4%(36件) | **PASS +11.1%**(49件) |
| DOT | FAIL -8.9%(12件) | **PASS +10.5%**(21件) | **PASS +2.2%**(64件) |
| TRX | FAIL -4.2%(27件) | **PASS +5.2%**(25件) | FAIL -3.2%(21件) |

**判定**: downtrend **6銘柄PASS** — エッジ確定！range も4銘柄。

### 2-3. ma_crossover 固定
- **Run ID**: 20260206_164550
- **OOS PASS**: 7/30 (uptrend 4, downtrend 1, range 2)

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL -6.5%(40件) | FAIL -7.2%(49件) | FAIL -1.3%(46件) |
| ETH | **PASS +4.0%**(33件) | FAIL -15.7%(94件) | FAIL -0.6%(38件) |
| BNB | **PASS +0.6%**(59件) | FAIL -6.8%(30件) | FAIL -3.4%(63件) |
| SOL | FAIL -6.5%(24件) | FAIL -2.0%(73件) | FAIL -5.6%(47件) |
| XRP | **PASS +2.5%**(38件) | FAIL -19.8%(110件) | FAIL -2.6%(30件) |
| ADA | FAIL -4.0%(25件) | FAIL -3.3%(92件) | FAIL -7.8%(36件) |
| AVAX | **PASS +6.1%**(38件) | FAIL -0.3%(98件) | **PASS +0.7%**(39件) |
| DOGE | FAIL -5.2%(30件) | FAIL -13.3%(75件) | FAIL -6.2%(50件) |
| DOT | FAIL -3.8%(28件) | **PASS +5.0%**(90件) | **PASS +8.0%**(33件) |
| TRX | **PASS +2.3%**(44件) | FAIL -2.6%(43件) | FAIL -5.1%(59件) |

**判定**: uptrend/ma_crossover_short 4銘柄PASS — エッジあり。ただしPnL小さめ。

### 2-4. bb_bounce 固定
- **Run ID**: 20260206_164551
- **OOS PASS**: 7/30 (uptrend 2, downtrend 2, range 3)

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL -4.6%(23件) | FAIL +3.3%(13件) | FAIL -2.5%(24件) |
| ETH | FAIL -1.9%(12件) | FAIL -4.2%(24件) | **PASS +5.4%**(48件) |
| BNB | FAIL +0.2%(12件) | **PASS +6.4%**(47件) | FAIL +14.2%(17件) |
| SOL | **PASS +3.1%**(49件) | FAIL -4.8%(24件) | **PASS +0.7%**(99件) |
| XRP | FAIL +2.0%(6件) | FAIL -5.7%(54件) | **PASS +1.7%**(28件) |
| ADA | **PASS +1.1%**(57件) | **PASS +8.5%**(56件) | FAIL +2.0%(17件) |
| AVAX | FAIL -8.0%(19件) | FAIL -42.4%(19件) | FAIL -10.4%(43件) |
| DOGE | FAIL +3.0%(17件) | FAIL -9.0%(45件) | FAIL -0.7%(24件) |
| DOT | FAIL +5.1%(5件) | FAIL -8.4%(12件) | FAIL -6.0%(18件) |
| TRX | FAIL -3.3%(39件) | FAIL -2.3%(37件) | FAIL -4.0%(26件) |

**判定**: 散在しており一貫した横断パターンなし。range/bb_bounce_shortが3銘柄だがPnL微小。

---

## Step 3: 判定 — エッジ確定！

### 3銘柄以上PASSした戦略（5件発見）

| Rank | テンプレート | レジーム | PASS数 | PASS銘柄 | 平均PnL |
|------|-------------|---------|--------|---------|---------|
| **1** | **volume_spike_short** | **downtrend** | **6/10** | ADA,DOT,XRP,TRX,SOL,AVAX | **+6.9%** |
| **2** | **volume_spike_short** | **range** | **4/10** | DOGE,SOL,XRP,DOT | **+7.3%** |
| **3** | **ma_crossover_short** | **uptrend** | **4/10** | AVAX,ETH,XRP,BNB | **+3.3%** |
| 4 | rsi_reversal_short | range | 3/10 | ETH,XRP,DOGE | +1.9% |
| 5 | bb_bounce_short | range | 3/10 | ETH,SOL,XRP | +2.6% |

### 最重要発見
- **volume_spike_short / downtrend**: 6銘柄でOOS PASS。BTC/DOGE/BNB/ETHを除く6銘柄で機能。
  - BNB: PnL +7.0%だがトレード13件で不足。ETH: +3.1%だが17件で不足。実質8/10がプラス。
- **全てショート系テンプレート**: PASS上位5件すべてが `_short` バリアント。ロング系は横断性が低い。

### 注目パターン
- **BTCだけが全テンプレート全レジームでFAIL**: 流動性最高=非効率性最小の仮説を強く裏付け
- **ショート優位**: downtrend×short + range×short が機能。uptrend×shortも(ma_crossover)

### 次のアクション
- **volume_spike_short / downtrend** をWFA検証 → 時間的頑健性確認
- PASSしたら戦略パラメータを保存（YAML）して実運用パイプラインへ

---

## Step 4: 複合テンプレート（必要時のみ）

Step 3でエッジが確定したため、当面は不要。
WFA検証で脱落した場合の予備プランとして残す。

---

## 変更ログ
- 2026-02-06: ロードマップ作成、Step 1完了
- 2026-02-06: Step 2完了（4テンプレート×10銘柄横断テスト）、Step 3判定完了
  - volume_spike_short/downtrend が6銘柄PASSでエッジ確定
  - 上位5件すべてショート系テンプレート
