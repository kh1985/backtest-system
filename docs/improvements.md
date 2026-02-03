# 改善案リスト

バックテストシステムの将来的な改善案を記録。

---

## レンジ検出の精度向上（案2）

**現状**: SwingStructure は HH/LL 混在パターンのみでレンジ判定（案1）

**改善案**: 構造崩れ + ATR縮小 の二重確認

```python
class SwingStructure(Indicator):
    def __init__(self, min_swing_bars=3, atr_filter=0.5, range_atr_ratio=0.8):
        self.range_atr_ratio = range_atr_ratio  # 追加パラメータ
        ...

    def calculate(self, df):
        # ATR縮小チェック（レンジ判定の補助）
        atr_sma = pd.Series(atr).rolling(20).mean().values

        for i in range(1, n):
            # 構造崩れ検出（現在の実装）
            structure_mixed = ...  # LL出現 or HH出現

            # ATR縮小検出
            volatility_contracted = atr[i] < atr_sma[i] * self.range_atr_ratio

            # レンジ判定: 両方満たす場合のみ
            if structure_mixed and volatility_contracted:
                current_trend = 0  # レンジ
            elif structure_mixed and not volatility_contracted:
                # 構造崩れてるがボラあり → トレンド転換の可能性
                pass  # 既存ロジックで転換判定
```

**期待効果**:
- 「深押し」と「本当のレンジ」を区別できる
- 誤検出（早すぎるレンジ判定）の削減

**実装タイミング**: 案1でバックテスト検証後、精度不足なら移行

---

## （その他の改善案をここに追記）

