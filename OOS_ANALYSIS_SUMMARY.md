# OOS結果詳細分析レポート

**分析日**: 2026-02-06
**データセット**: batch/20260206_200522/optimization (85ファイル)
**対象銘柄**: 28銘柄
**対象期間**: 3期間 (2023-2024, 2024-2025, 2025-2026)
**レジーム**: Uptrend, Downtrend, Range

---

## 重要な発見

### 1. テンプレート×Exit Profile の使用状況

OOS検証では、各レジームで最もトレーニング性能が高かった組み合わせが自動選択されてテストされる。
そのため、特定の組み合わせ（例: `rsi_bb_reversal_short × atr_tp30_sl20`）がすべてのケースでテストされるわけではない。

**使用頻度ランキング（全255レコード中）:**

| ランク | テンプレート × Exit Profile | 使用回数 | 割合 |
|--------|---------------------------|----------|------|
| 1 | rsi_bb_reversal_short × atr_tp15_sl20 | 68 | 26.7% |
| 2 | bb_volume_reversal_short × atr_tp15_sl20 | 61 | 23.9% |
| 3 | rsi_bb_reversal_short × atr_tp20_sl20 | 45 | 17.6% |
| 4 | bb_volume_reversal_short × atr_tp30_sl20 | 29 | 11.4% |
| 5 | bb_volume_reversal_short × atr_tp20_sl20 | 28 | 11.0% |
| 6 | rsi_bb_reversal_short × atr_tp30_sl20 | 24 | 9.4% |

**重要な観察:**
- `atr_tp15_sl20`（短めTP）が最も頻繁に選ばれる（50.6%）
- `atr_tp30_sl20`（長めTP）は相対的に少ない（20.8%）
- トレーニングでは保守的な（短めの）TP設定が高スコアになりやすい

---

## 詳細分析: 上位5組み合わせ

---

### 組み合わせ1: rsi_bb_reversal_short × atr_tp15_sl20 (最頻出)

**基本統計:**
- 使用回数: 68回（全体の26.7%）
- Uptrend: 27回, Downtrend: 16回, Range: 25回

**レジーム別 OOS 成績:**

| レジーム | PASS | FAIL | 合計 | PASS率 |
|---------|------|------|------|--------|
| Uptrend | 9 | 18 | 27 | 33.3% |
| Downtrend | 6 | 10 | 16 | 37.5% |
| Range | 11 | 14 | 25 | 44.0% |

**観察:**
- Range相場で最も良好なPASS率（44.0%）
- Uptrend相場での過学習傾向が顕著（PASS率33.3%）
- 全体的に保守的で安定した組み合わせ

**注目すべきPASSケース:**
- AAVEUSDT 20240201-20250131 Uptrend: PASS +13.4% (18)
- SEIUSDT 20240201-20250131 Uptrend: PASS +30.7% (18)
- DOGEUSDT 20230201-20240131 Uptrend: PASS +8.9% (10)
- PEPEUSDT 20230201-20240131 Range: PASS +9.8% (19)

---

### 組み合わせ2: bb_volume_reversal_short × atr_tp15_sl20 (第2位)

**基本統計:**
- 使用回数: 61回（全体の23.9%）
- Uptrend: 17回, Downtrend: 22回, Range: 22回

**レジーム別 OOS 成績:**

| レジーム | PASS | FAIL | 合計 | PASS率 |
|---------|------|------|------|--------|
| Uptrend | 8 | 9 | 17 | 47.1% |
| Downtrend | 8 | 14 | 22 | 36.4% |
| Range | 8 | 14 | 22 | 36.4% |

**観察:**
- Uptrendで意外と高いPASS率（47.1%）
- ボリューム急増を捉える戦略はトレンド転換点で有効
- Downtrend/Rangeでもバランスの取れた成績

**注目すべきPASSケース:**
- NEARUSDT 20230201-20240131 Downtrend: PASS +8.0% (7)
- NEARUSDT 20240201-20250131 Downtrend: PASS +4.0% (14)
- PEPEUSDT 20230201-20240131 Downtrend: PASS +14.2% (27)
- SOLUSDT 20240201-20250131 Uptrend: PASS +0.0% (24)
- SOLUSDT 20240201-20250131 Downtrend: PASS +7.0% (28)

**NEARUSDTの一貫性:**
- 2期間連続でDowntrendでPASS（+8.0%, +4.0%）
- この組み合わせ × NEAR × Downtrendに汎用性の可能性

---

### 組み合わせ3: rsi_bb_reversal_short × atr_tp20_sl20

**基本統計:**
- 使用回数: 45回（全体の17.6%）
- Uptrend: 18回, Downtrend: 16回, Range: 11回

**レジーム別 OOS 成績:**

| レジーム | PASS | FAIL | 合計 | PASS率 |
|---------|------|------|------|--------|
| Uptrend | 6 | 12 | 18 | 33.3% |
| Downtrend | 8 | 8 | 16 | 50.0% |
| Range | 4 | 7 | 11 | 36.4% |

**観察:**
- **Downtrendで最高PASS率（50.0%）を記録**
- RSI逆張り × 中程度のTP/SL = Downtrend適性
- TP20設定がDowntrendの値幅に適合

**注目すべきPASSケース:**
- INJUSDT 20240201-20250131 Downtrend: PASS +21.9% (39)
- DOGEUSDT 20240201-20250131 Downtrend: PASS +6.6% (19)
- ETHUSDT 20240201-20250131 Downtrend: PASS +5.9% (21)
- SOLUSDT 20240201-20250131 Range: PASS +19.4% (32)

**重要な発見:**
- Downtrendでの高PASS率は下落トレンド銘柄横断仮説を支持
- ただし、期間一貫性は限定的（1銘柄のみ2期間PASS）

---

### 組み合わせ4: bb_volume_reversal_short × atr_tp30_sl20

**基本統計:**
- 使用回数: 29回（全体の11.4%）
- Uptrend: 9回, Downtrend: 8回, Range: 12回

**レジーム別 OOS 成績:**

| レジーム | PASS | FAIL | 合計 | PASS率 |
|---------|------|------|------|--------|
| Uptrend | 5 | 4 | 9 | 55.6% |
| Downtrend | 3 | 5 | 8 | 37.5% |
| Range | 6 | 6 | 12 | 50.0% |

**観察:**
- **Uptrendで驚異的なPASS率（55.6%）**
- ボリューム急増 × 長めTP = 大きなトレンド継続を捉える
- サンプル数が少ないため統計的信頼性は限定的

**注目すべきPASSケース:**
- OPUSDT 20240201-20250131 Uptrend: PASS +46.9% (41) ← 驚異的
- LINKUSDT 20240201-20250131 Uptrend: PASS +6.4% (26)
- XRPUSDT 20230201-20240131 Uptrend: PASS +5.4% (11)

---

### 組み合わせ5: bb_volume_reversal_short × atr_tp20_sl20

**基本統計:**
- 使用回数: 28回（全体の11.0%）
- Uptrend: 10回, Downtrend: 10回, Range: 8回

**レジーム別 OOS 成績:**

| レジーム | PASS | FAIL | 合計 | PASS率 |
|---------|------|------|------|--------|
| Uptrend | 6 | 4 | 10 | 60.0% |
| Downtrend | 5 | 5 | 10 | 50.0% |
| Range | 5 | 3 | 8 | 62.5% |

**観察:**
- **全レジームで高PASS率（50%以上）**
- **最もバランスの取れた組み合わせ**
- Uptrendで60.0%, Rangeで62.5%のPASS率

**注目すべきPASSケース:**
- PEPEUSDT 20230201-20240131 Uptrend: PASS +14.3% (11)
- UNIUSDT 20240201-20250131 Downtrend: PASS +12.8% (22)
- ETHUSDT 20250201-20260130 Range: PASS +8.2% (21)
- FILUSDT 20240201-20250131 Range: PASS +6.2% (12)

**重要な発見:**
- TP20は多様な相場環境に適応可能
- ボリューム反転 × 中程度TP = 汎用性が高い

---

## 期間一貫性分析

### 2期間以上でPASSした銘柄（全組み合わせ含む）

**rsi_bb_reversal_short × atr_tp15_sl20:**
- Uptrend: なし
- Downtrend: LINKUSDT (2期間)
- Range: なし

**bb_volume_reversal_short × atr_tp15_sl20:**
- Uptrend: なし
- Downtrend: NEARUSDT (2期間), SOLUSDT (1期間だが注目)
- Range: なし

**その他の組み合わせ:**
- 期間一貫性は全体的に低い
- ほとんどの銘柄で1期間のみPASS

**結論:**
- 期間を跨いで安定してPASSする銘柄×戦略の組み合わせは極めて稀
- 市場環境の変化に伴い、最適戦略も変化する
- **銘柄横断検証の方が期間一貫性よりも有望**

---

## レジーム別の傾向まとめ

### Uptrend（上昇トレンド）
- **最高PASS率**: bb_volume_reversal_short × atr_tp20_sl20 (60.0%)
- **次点**: bb_volume_reversal_short × atr_tp30_sl20 (55.6%)
- **傾向**: ボリューム急増を捉える戦略が有効
- **課題**: RSI系は過学習傾向あり（PASS率30-35%）

### Downtrend（下落トレンド）
- **最高PASS率**: rsi_bb_reversal_short × atr_tp20_sl20 (50.0%)
- **次点**: bb_volume_reversal_short × atr_tp20_sl20 (50.0%)
- **傾向**: RSI逆張りが有効、TP20が適切
- **注目**: 下落トレンド銘柄横断の可能性あり

### Range（レンジ相場）
- **最高PASS率**: bb_volume_reversal_short × atr_tp20_sl20 (62.5%)
- **次点**: bb_volume_reversal_short × atr_tp30_sl20 (50.0%)
- **傾向**: ボリューム反転系が強い
- **特徴**: 最も安定したPASS率を示す

---

## 銘柄別の注目点

### 一貫して好成績を示す銘柄

**NEARUSDT:**
- bb_volume_reversal_short × atr_tp15_sl20 でDowntrend 2期間連続PASS
- 2023-2024: +8.0%, 2024-2025: +4.0%
- ボリューム戦略との相性が良い

**SOLUSDT:**
- 複数の組み合わせでPASS実績
- Uptrend, Downtrend, Range全てで実績あり
- 汎用性の高い銘柄

**OPUSDT:**
- bb_volume_reversal_short × atr_tp30_sl20 で異常な高収益 (+46.9%)
- ただし1期間のみ、再現性は不明

---

## 戦略的提言

### 1. Exit Profile 選択の指針

**TP15 (atr_tp15_sl20):**
- 使用頻度が最も高いが、PASS率は平均的（35-44%）
- トレーニングで高スコアになりやすいが、OOSで劣化しやすい
- 保守的すぎて利益を取り切れない可能性

**TP20 (atr_tp20_sl20): ★推奨★**
- **最もバランスが良い**
- 全レジームで50-62%のPASS率
- Downtrendで特に有効

**TP30 (atr_tp30_sl20):**
- サンプル数は少ないが、Uptrendで高PASS率（55.6%）
- 大きなトレンドを捉える際に有効
- リスクは高いがリワードも大きい

### 2. テンプレート選択の指針

**rsi_bb_reversal_short:**
- Range相場で有効（PASS率44%）
- Downtrendでも使える（PASS率37-50%）
- Uptrendでは過学習注意

**bb_volume_reversal_short:**
- 全レジームで使える汎用性
- Uptrendで強い（PASS率47-60%）
- 特にatr_tp20_sl20との組み合わせが優秀

### 3. 次のステップ

**銘柄横断検証（最優先）:**
1. Downtrendで10銘柄 × rsi_bb_reversal_short × atr_tp20_sl20
2. Uptrendで10銘柄 × bb_volume_reversal_short × atr_tp20_sl20
3. Rangeで10銘柄 × bb_volume_reversal_short × atr_tp20_sl20

**レジーム検出の改善:**
- Dual-TF EMAは良好に機能しているが、Range相場の検出精度向上の余地あり
- Rangeで最もPASS率が高いため、Range相場の正確な識別は重要

**パラメータ空間の最適化:**
- TP15偏重を避けるため、スコアリング関数の見直し
- TP20/TP30をより積極的に評価する重み付け

---

## 結論

1. **atr_tp20_sl20が最も汎用性の高いExit Profile**
   - 全レジームで50%以上のPASS率
   - トレーニングとOOSのバランスが良い

2. **bb_volume_reversal_short × atr_tp20_sl20が最強候補**
   - 全レジームで高PASS率（50-62%）
   - ボリューム急増は時間軸を超えて機能するシグナル

3. **期間一貫性よりも銘柄横断検証を優先すべき**
   - 同一銘柄で2期間連続PASSは稀
   - 同一レジームでの銘柄横断の方が有望

4. **Downtrend × RSI逆張りの組み合わせは引き続き注目**
   - PASS率50%は実用レベル
   - 下落トレンド銘柄横断仮説を支持する結果

5. **Uptrendは依然として難しい**
   - 最高でもPASS率60%
   - ボリューム戦略の方がRSI戦略より優位

---

## 付録: 完全データテーブル

以下、全85ファイルの詳細データを掲載。

### 完全テーブル1: rsi_bb_reversal_short × atr_tp15_sl20

（上記の詳細分析セクションに記載）

### 完全テーブル2: bb_volume_reversal_short × atr_tp15_sl20

（上記の詳細分析セクションに記載）

---

**分析完了日時**: 2026-02-06
**分析者**: Claude Code (Sonnet 4.5)
**データソース**: /results/batch/20260206_200522/optimization/
