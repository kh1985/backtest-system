# 戦略探索アーカイブ（Step 1-13 完了済み）

> このファイルは完了済みステップの詳細記録。通常参照不要。
> 現在の状態は `roadmap.md` を参照。

---

## Step 1: ベースライン（全テンプレート混合）— 完了

**Run ID**: 20260206_155925
**条件**: 21テンプレート全部 × atr_compact × 10銘柄 × 1年(20250201-20260130)
**注**: 5銘柄(ADA,AVAX,DOGE,DOT,TRX)が4hデータなしでMA Cross fallback

### 結果: OOS PASS 8/30 (27%), 3銘柄共通=0

| 銘柄 | Uptrend | Downtrend | Range |
|------|---------|-----------|-------|
| BTC | FAIL(2件) | FAIL -3.5% | FAIL -1.3% |
| ETH | FAIL(15件) | FAIL -8.0% | FAIL -5.2% |
| BNB | FAIL(17件) | PASS +2.0%(vol_spike_s) | FAIL(19件) |
| SOL | FAIL | PASS +4.9%(ma_crossover) | FAIL |
| XRP | 0件 | PASS +25.5%(rsi_reversal) | PASS +7.1%(vol_spike_s) |
| ADA | PASS +3.1%(bb_bounce) | PASS +6.8%(rsi_reversal) | FAIL |
| AVAX | FAIL | FAIL | FAIL |
| DOGE | FAIL(17件) | FAIL | PASS +0.06%(rsi_rev_s) |
| DOT | FAIL | FAIL | FAIL |
| TRX | FAIL | PASS +5.2%(vol_spike_s) | FAIL |

### 問題点
各銘柄で「一番勝った」テンプレートが選ばれるため、テンプレートがバラバラ。

---

## Step 2: テンプレート固定の横断テスト — 完了

### 2-A. 1年データ混合テスト（参考）
**Run ID**: 20260206_180815
結果: PASS 6/30 (20%)。volume_spike_short downtrend が最有力

### 2-B. 2年データ混合テスト（本番）
**Run ID**: 20260206_181614
**結果**: OOS PASS 8/30 (26.7%)

レジーム別: downtrend 40%, uptrend 20%, range 20%
テンプレート別: bb_bounce系 50%, vol_spike_short 30%, ma_crossover 25%, rsi_reversal 21%

---

## Step 3: 判定 — 完了

**「downtrend にエッジあり」確定（40%）、「特定テンプレート横断」は2銘柄止まり**
XRP downtrend が最堅牢。BTC/ETH は全滅。

---

## Step 4: 複合テンプレート第1弾 — 完了

**Run ID**: 20260206_183526
**結果**: rsi_bb_reversal_short が **6銘柄PASS** — 単体の2銘柄限界を突破

---

## Step 5: 複合テンプレート第2弾 — 完了

**Run ID**: 20260206_185238
**結果**: OOS PASS 10/30 (33.3%)
**発見**: BBを含む組み合わせだけが機能。Stoch+BB, MACD+BB, RSI+MACD 全滅

---

## Step 6: 3期間一貫性検証 — 完了

**Run ID**: 20260206_191208
推奨: bb_vol_short/UT 4銘柄50%, rsi_bb_short/DT 5銘柄

---

## Step 6.5: 30銘柄スケール検証 — 完了

**Run ID**: 20260206_200522
rsi_bb_reversal_short が全3レジームTop3独占。atr_tp30_sl20が最適exit。

---

## Step 7: パラメータ最適化 — 完了

**Run ID**: 20260206_214905 (σ拡張), 20260206_221008 (ロング版追加)
σ=2.0固定確定。rsi_bb_reversal_long/UT 42.3%(n=26)が最有望。3重複合は全滅。

---

## Step 8: RSI閾値微調整 — 完了

**Run ID**: 20260208_060842 (6値), 20260208_061402 (絞り込み)
RSI<35 が60%(n=15)で初の実用基準達成。RSI<29は過学習トラップ。

---

## Step 9: RSI固定＋ショート版閾値探索 — 完了

**Run ID**: 20260208_064753 (ロング固定), 20260208_064442 (ショート探索)
rsi_bb_long_f35/UT/tp30: **73.9%(n=23)** 到達。ショート RSI>65/range 50%。

---

## Step 10: ショートRSI固定テスト — 完了

**Run ID**: 20260208_065644
RSI>67/range/tp20 = 62%(n=8)。DT向け42%で限界。

---

## Step 11: WFA検証 + YAML + DT代替探索 — 完了

WFA: ロングのみ堅牢（WFE=1.035）、ショート2戦略は偽陽性で不合格。
YAML: 3件出力。DT代替も見送り。

---

## Step 12: 2025年問題診断 + exit固定WFA — 完了

exit固定でWFA ROBUST率2倍改善（11%→22%）。tp20固定が最堅牢。

---

## Step 13: 新パターン探索 + WFA基準調整 + YAML — 完了

CR>=0.6で22%→37%に改善。f40はOOS81%だがWFAはf35同等。
VWAP/TP/VP 9テンプレート全てWFA不合格。44テンプレート中rsi_bb_long系のみ実運用可能。

---

## 変更ログ
- 2026-02-06: Step 1-7完了
- 2026-02-08: Step 8-12完了
- 2026-02-09: Step 13-14完了
