# Agent Progress（自律ループ用）

## 状態
- status: idle
- last_iteration: 1
- last_completed_task: ADX+BB探索 — 新実運用戦略adx_bb_long/tp30発見
- consecutive_failures: 0（最終的に成功）

## 完了タスク

### Iteration 1: ADX+BB系探索（2026-02-09）

**スカウトテスト:**
| # | テンプレート | レジーム | OOS PASS率 | 判定 |
|---|---|---|---|---|
| 1 | adx_bb_long + adx_bb_short | UT/DT | UT:43%/DT:25% | UT有望→フル検証へ |
| 2 | ema_fast_cross_bb_long + ema_fast_bb_lower_long | - | 0% | 全滅（crossover条件でトレード不足） |
| 3 | ema_state_bb_lower_long + ema_state_bb_mid_long | UT | UT:-5.8%PnL | UT向けPnLマイナス |
| 4 | di_bb_lower_long + di_bb_upper_short | - | 0% | 全滅 |

**フル検証（adx_bb_long × 26銘柄）:**
- UT/tp30: OOS 30%, +6.8%
- UT/tp20: OOS 27%, +4.2%

**WFA検証（adx_bb_long × 18銘柄）:**
| Exit | ROBUST率 | 平均PnL |
|---|---|---|
| **tp30** | **37% (20/54)** | **+10.4%** |
| tp20 | 28% (15/54) | +9.8% |

**★ 発見: adx_bb_long/uptrend/tp30 = ROBUST 37%、rsi_bb_long_f35/tp20と同率！**

**ROBUST銘柄ランク（adx_bb_long/tp30, CR>=0.6）:**
| 銘柄 | ROBUST/3 | 平均PnL |
|---|---|---|
| **NEARUSDT** | **3/3** | **+34.9%** |
| FILUSDT | 2/3 | +35.0% |
| SUIUSDT | 2/3 | +31.5% |
| AVAXUSDT | 2/3 | +24.0% |

**教訓:**
1. crossover（瞬間）条件はロング側ではトレード不足になる
2. EMAモメンタム確認（状態型）はUT向けではPnLマイナス
3. +DI/-DI方向フィルタはBB条件との組み合わせで全滅
4. ADX（トレンド強度）+ BB lower = 有効な新しいエッジ
5. ADXはtp30（大きめの利幅）で最も安定

### 追加テンプレート（#62-#69）
| # | テンプレート | 結果 |
|---|---|---|
| 62 | adx_bb_long | ★ WFA 37% → 実運用候補 |
| 63 | adx_bb_short | OOS境界（DT 25-50%）だがlongほどではない |
| 64 | ema_fast_cross_bb_long | 全滅（crossoverでトレード不足） |
| 65 | ema_fast_bb_lower_long | 全滅（crossover版） |
| 66 | ema_state_bb_lower_long | 全滅（BB lowerでもトレード不足） |
| 67 | ema_state_bb_mid_long | UT向けPnLマイナス |
| 68 | di_bb_lower_long | 全滅 |
| 69 | di_bb_upper_short | 全滅 |

## 残タスク
1. adx_bb_long/uptrend/tp30の確定戦略YAMLを作成
2. rsi_bb_long_f35 vs adx_bb_long のROBUST銘柄の重複度分析（相互補完性の定量評価）
3. DT戦略YAML作成: ema_fast_cross_bb_short / downtrend / tp15
4. UT+DT複合運用シミュレーション: 3戦略（rsi_bb + adx_bb + ema_fast_cross_bb）同時運用の推定
