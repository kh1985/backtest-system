# トラリピ（トラップリピートイフダン）設計書

## 1. 概要

トラリピは価格レンジ内に複数の買い（または売り）注文を等間隔で配置し、約定時に指定利幅でリピート注文を出すグリッド型自動売買戦略。

**特徴:**
- レンジ相場で威力を発揮（トレンド相場では含み損拡大のリスク）
- 価格が上下動する度に利益を積み重ねる
- 資金管理が重要（複数ポジション同時保有）

## 2. 基本ロジック

### 2.1 コアパラメータ

| パラメータ | 説明 | 例 |
|-----------|------|---|
| **range_low** | レンジ下限価格 | 30000 USDT |
| **range_high** | レンジ上限価格 | 40000 USDT |
| **trap_interval** | トラップ間隔（USD） | 500 USDT |
| **profit_width** | 利確幅（USD） | 300 USDT |
| **side** | ロング/ショート | "long" |
| **max_positions** | 最大同時保有数 | 10 |

### 2.2 動作フロー（ロングの場合）

```
1. レンジ内にトラップ配置
   - range_low から range_high まで trap_interval ごとに買い注文
   - 例: 30000, 30500, 31000, ..., 40000

2. 価格がトラップに到達
   - 該当価格でポジションエントリー
   - entry_price = trap_price

3. 利確
   - TP価格 = entry_price + profit_width
   - 例: 30000で買い → 30300でTP

4. リピート
   - 決済後、同じトラップ価格で再度買い注文を待機
   - レンジ内を価格が上下する限り自動で繰り返す
```

### 2.3 ショートの場合

- レンジ上限から下限に向けて売り注文配置
- TP価格 = entry_price - profit_width
- 価格上昇で利確、下落でエントリー

## 3. OHLCVデータでの実装方法

### 3.1 課題: 複数ポジション管理

Prismの現行エンジンは **1ポジションモデル** (`BacktestEngine.run()` 内で `position: Optional[Position] = None`)。トラリピは同時に複数のトラップで約定する可能性があるため、以下の実装オプションがある。

#### オプションA: 簡易版（1ポジションに集約）

**実装:**
- 各バーで、レンジ内で最も近いトラップ価格を1つだけエントリー
- TP決済後、同じトラップで再エントリー可能
- `max_positions=1` として動作

**メリット:**
- 既存エンジン (`BacktestEngine`) を改修不要
- テンプレート (`StrategyTemplate`) として実装可能

**デメリット:**
- 実際のトラリピと乖離（複数ポジション同時保有を再現できない）
- 含み損の積み重ねシミュレーション不可

#### オプションB: 完全版（複数ポジション管理）

**実装:**
- `BacktestEngine` に複数ポジション管理機能を追加
- `positions: List[Position]` で複数ポジションを保持
- 資金管理: `initial_capital` をポジション数に応じて分割
- 各トラップごとにポジションサイズを計算

**メリット:**
- 実際のトラリピを正確にシミュレーション
- レンジ相場でのリスク（含み損積み重ね）を正しく評価

**デメリット:**
- エンジン改修が必要（`engine/backtest.py` の大幅変更）
- 資金管理ロジックが複雑化

**推奨: オプションA → オプションB の段階的実装**
1. まずオプションAで簡易版テンプレートを作成し、コンセプトを検証
2. 有効性が確認できたらオプションBで完全版エンジンを実装

### 3.2 エントリー条件（オプションA）

```python
# トラップ配置
traps = np.arange(range_low, range_high + trap_interval, trap_interval)

# 各バーでトラップとの距離をチェック
def check_trap_entry(row, prev_row):
    close = row["close"]
    low = row["low"]
    high = row["high"]

    for trap_price in traps:
        if side == "long":
            # 価格がトラップを下から上に通過（または接触）
            if prev_row["close"] < trap_price <= close:
                return Signal(side="long", entry_price=trap_price)
            # bar内でトラップをまたいだ場合
            if low <= trap_price <= high:
                return Signal(side="long", entry_price=trap_price)
        else:  # short
            # 価格がトラップを上から下に通過
            if prev_row["close"] > trap_price >= close:
                return Signal(side="short", entry_price=trap_price)
            if low <= trap_price <= high:
                return Signal(side="short", entry_price=trap_price)

    return None
```

### 3.3 決済条件

```python
# ロングの場合
tp_price = entry_price + profit_width
sl_price = None  # トラリピは基本的にSL無し（レンジ逸脱リスクあり）

# ショートの場合
tp_price = entry_price - profit_width
```

**リスク管理オプション:**
- レンジ逸脱時のSL設定（例: range_low * 0.95）
- タイムアウト設定（例: 100バー保有でTIMEOUT決済）

### 3.4 リピート機能

- 既存の `BacktestEngine` は決済後に `position = None` となり、次のバーで再度エントリー判定が可能
- トラップ価格は固定なので、価格がレンジ内で往復すれば自動でリピートされる

## 4. レンジ検出アルゴリズム

トラリピの成否はレンジ検出の精度に依存。以下の方法を提案。

### 4.1 方法1: 固定レンジ（手動設定）

**実装:**
- ユーザーが `range_low` と `range_high` を直接指定
- 最もシンプルだが、レンジ逸脱リスクが高い

**適用場面:**
- 歴史的価格帯が明確（例: ステーブルコインのペッグ付近）
- レンジ相場が長期間継続することが確実な場合

### 4.2 方法2: ATRベース動的レンジ

**実装:**
```python
from indicators.volatility import ATR

# 現在価格 ± ATR倍率でレンジ設定
atr = calculate_atr(df, period=14)
center_price = df["close"].iloc[-1]
range_width = atr * 2.0  # ATRの2倍を全幅とする

range_low = center_price - range_width / 2
range_high = center_price + range_width / 2
```

**メリット:**
- ボラティリティに応じて自動調整
- トレンド転換時にレンジを再設定可能

**デメリット:**
- レンジが固定されず、トラップ価格が変動する（リピート性が低下）

### 4.3 方法3: ボリンジャーバンドベースレンジ

**実装:**
```python
# BB(20, 2σ) の上限/下限をレンジとする
bb_upper = df["bb_upper"]
bb_lower = df["bb_lower"]

range_low = bb_lower.iloc[-1]
range_high = bb_upper.iloc[-1]
```

**メリット:**
- 価格分布の統計的性質を反映
- レンジ相場では安定

**デメリット:**
- トレンド相場ではバンドが一方向に移動し、トラリピが機能しない

### 4.4 方法4: Volume Profile POC ± 範囲

**実装:**
```python
# Volume ProfileのPOC（最大出来高価格帯）を中心に設定
from indicators.structure import VolumeProfile

vp = VolumeProfile(num_bins=50)
df = vp.calculate(df)
poc_price = df["vp_poc"].iloc[-1]

# POC ± 5% をレンジとする
range_low = poc_price * 0.95
range_high = poc_price * 1.05
```

**メリット:**
- 市場参加者の集中価格帯を反映
- レンジ相場でPOC付近に価格が戻りやすい性質を利用

**デメリット:**
- POC計算にデータ期間が必要（短期間では不安定）

### 4.5 推奨: Dual-TF EMA + BB組み合わせ

Prismの既存レジーム検出（`analysis/trend.py`）を活用し、**Rangeレジームのみでトラリピを発動**。

```python
# 1. Dual-TF EMAでレンジ判定
regime = detect_dual_tf_ema_regime(df)
is_range = (regime == "range")

# 2. レンジ期間のBB範囲をトラップ配置範囲とする
if is_range:
    range_low = df["bb_lower"].iloc[-1]
    range_high = df["bb_upper"].iloc[-1]
    traps = calculate_traps(range_low, range_high, trap_interval)
```

**メリット:**
- トレンド相場でトラリピを起動しない（含み損拡大を回避）
- レンジ判定の精度が高い（Dual-TF EMAは既にWFA検証済み）
- BBでレンジ幅が自動調整される

## 5. バックテスト検証方法

### 5.1 検証指標

| 指標 | 重要度 | 目標値 |
|------|--------|--------|
| **総利益** | ★★★ | プラス |
| **勝率** | ★★☆ | 70%以上（小さい利幅を繰り返すため） |
| **最大ドローダウン** | ★★★ | <20%（レンジ逸脱リスク） |
| **トレード数** | ★★★ | 最低30件（統計的有意性） |
| **平均保有期間** | ★☆☆ | 短いほど良い（回転率重視） |
| **含み損（最大）** | ★★★ | 資金管理上の最重要指標 |

### 5.2 検証フロー

```bash
# Step 1: 簡易版テンプレート実装（オプションA）
# optimizer/templates.py に trap_repeat_long/short を追加

# Step 2: OOS検証
python3 -m modal run scripts/modal_optimize.py \
  --templates trap_repeat_long trap_repeat_short \
  --regimes range \
  --exit-profiles fixed

# Step 3: WFA検証（5フォールドAnchored Walk-Forward）
python scripts/local_wfa_test.py \
  --result-json results/optimization/run_YYYYMMDD_HHMMSS.json \
  --template trap_repeat_long \
  --regimes range \
  --cr-threshold 0.6

# Step 4: レジーム別分析
# - Range期間のみでROBUST率を評価
# - Uptrend/Downtrend期間では動作しないことを確認
```

### 5.3 リスクシナリオ検証

| シナリオ | 検証方法 |
|----------|---------|
| **レンジ逸脱（ブレイクアウト）** | トレンド転換時のドローダウンを測定。SL設定の有無で比較。 |
| **ボラティリティ急騰** | 2020年3月、2022年5-6月の暴落期間でバックテスト。 |
| **トレード機会不足** | レンジ幅が狭すぎる（trap_interval > range_width）場合の挙動確認。 |
| **資金枯渇** | 複数ポジション同時保有（オプションB）で資金が不足する条件を特定。 |

### 5.4 パラメータグリッドサーチ

```python
ParameterRange("trap_interval", min_val=0.5, max_val=2.0, step=0.5, param_type="float")  # %
ParameterRange("profit_width", min_val=0.3, max_val=1.0, step=0.1, param_type="float")  # %
ParameterRange("max_positions", min_val=5, max_val=20, step=5, param_type="int")
```

**探索空間サイズ:** 4 × 8 × 4 = 128 configs（許容範囲）

## 6. 既存テンプレートとの統合案

### 6.1 テンプレート定義（optimizer/templates.py）

```python
@dataclass
class TrapRepeatTemplate(StrategyTemplate):
    """
    トラリピ（トラップリピートイフダン）テンプレート

    レンジ相場専用。価格レンジ内に等間隔でトラップを配置し、
    約定時に固定利幅で決済後、同一トラップで再エントリーを繰り返す。
    """

    name: str = "trap_repeat_long"
    description: str = "トラリピ（ロング）: BB範囲内にトラップ配置"

    config_template = {
        "name": "trap_repeat_long",
        "side": "long",
        "indicators": [
            {"type": "bollinger_bands", "period": 20, "std_dev": 2.0},
        ],
        "entry_conditions": [
            {
                "type": "trap_grid",  # 新規条件タイプ（後述）
                "trap_interval_pct": "{trap_interval}",  # %
                "range_source": "bb",  # "bb" or "fixed" or "atr"
            }
        ],
        "exit": {
            "take_profit_pct": "{profit_width}",
            "stop_loss_pct": 0.0,  # デフォルトSL無し
            "timeout_bars": 100,  # レンジ長期滞留対策
        }
    }

    param_ranges = [
        ParameterRange("trap_interval", 0.5, 2.0, 0.5, "float"),
        ParameterRange("profit_width", 0.3, 1.0, 0.1, "float"),
    ]


# ショート版
trap_repeat_short = TrapRepeatTemplate(
    name="trap_repeat_short",
    description="トラリピ（ショート）: BB範囲内にトラップ配置",
    config_template={
        **trap_repeat_long.config_template,
        "side": "short",
    }
)
```

### 6.2 新規Conditionクラス（strategy/conditions.py）

```python
class TrapGridCondition(Condition):
    """
    トラップグリッド条件

    レンジ内に等間隔でトラップを配置し、価格が接触したらエントリー。
    """

    def __init__(
        self,
        trap_interval_pct: float,
        range_source: str = "bb",  # "bb", "fixed", "atr"
        range_low: Optional[float] = None,  # range_source="fixed"用
        range_high: Optional[float] = None,
        side: str = "long",
    ):
        self.trap_interval_pct = trap_interval_pct
        self.range_source = range_source
        self.range_low = range_low
        self.range_high = range_high
        self.side = side
        self.traps = []

    def _calculate_traps(self, row: pd.Series) -> List[float]:
        """現在のレンジ範囲からトラップ価格リストを計算"""
        if self.range_source == "bb":
            low = row.get("bb_lower", 0)
            high = row.get("bb_upper", 0)
        elif self.range_source == "fixed":
            low = self.range_low
            high = self.range_high
        elif self.range_source == "atr":
            center = row["close"]
            atr = row.get("atr", 0)
            width = atr * 2.0
            low = center - width / 2
            high = center + width / 2
        else:
            return []

        if low >= high:
            return []

        interval = (high - low) * self.trap_interval_pct / 100
        traps = np.arange(low, high + interval, interval)
        return traps.tolist()

    def evaluate(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """価格がいずれかのトラップに接触したか判定"""
        self.traps = self._calculate_traps(row)

        close = row["close"]
        low = row["low"]
        high = row["high"]
        prev_close = prev_row["close"]

        for trap in self.traps:
            if self.side == "long":
                # 価格がトラップを下から上に通過、またはbar内で接触
                if prev_close < trap <= close:
                    return True
                if low <= trap <= high:
                    return True
            else:  # short
                # 価格がトラップを上から下に通過
                if prev_close > trap >= close:
                    return True
                if low <= trap <= high:
                    return True

        return False

    def describe(self) -> str:
        return f"TrapGrid({self.range_source}, interval={self.trap_interval_pct}%)"
```

### 6.3 統合手順

1. **新規Condition追加**: `strategy/conditions.py` に `TrapGridCondition` を実装
2. **ConditionBuilder拡張**: `strategy/builder.py` の `_build_condition()` に `"trap_grid"` タイプを追加
3. **テンプレート登録**: `optimizer/templates.py` の `get_all_templates()` に `trap_repeat_long/short` を追加
4. **レジーム制限**: `optimizer/grid_search.py` で `trap_repeat_*` は `regimes=["range"]` のみ許可

## 7. 実装優先順位

### フェーズ1: 簡易版プロトタイプ（オプションA）
- [ ] `TrapGridCondition` 実装（BBベースレンジ固定）
- [ ] `trap_repeat_long/short` テンプレート追加
- [ ] 単一銘柄でローカルバックテスト実行
- [ ] パラメータ感度テスト（trap_interval × profit_width）

**期待成果:** トラリピのコンセプト検証。レンジ相場での有効性を確認。

### フェーズ2: OOS/WFA検証
- [ ] Modal環境で30銘柄×3期間のOOS検証
- [ ] Range期間のみでWFA実行
- [ ] ROBUST率30%以上ならフェーズ3へ進む

**合格基準:** WFA CR >= 0.6、ROBUST率 >= 30%（Range期間のみ）

### フェーズ3: 完全版エンジン（オプションB）
- [ ] `BacktestEngine` を複数ポジション対応に改修
- [ ] `positions: List[Position]` での資金管理実装
- [ ] 含み損・証拠金維持率の可視化
- [ ] レンジ逸脱時のSL機能追加

**期待成果:** 実運用レベルのトラリピシミュレーター完成。

### フェーズ4: レンジ検出最適化
- [ ] ATRベース動的レンジ実装
- [ ] Volume Profile POCベースレンジ実装
- [ ] レンジ検出精度の比較検証

## 8. リスクと制約事項

| リスク | 対策 |
|--------|------|
| **レンジ逸脱（トレンド発生）** | Dual-TF EMAで事前にレンジ判定。トレンド期間は稼働停止。 |
| **含み損の積み重ね** | max_positions制限。SL設定オプション。 |
| **トレード機会不足** | trap_interval を適切に設定。BB幅が狭すぎる場合はエントリー見送り。 |
| **資金管理の複雑化** | フェーズ1では簡易版で検証。フェーズ3で完全版実装。 |
| **バックテストの限界** | OHLCVデータでは厳密なfill価格を再現できない。実運用前にデモ取引推奨。 |

## 9. 成功指標（KPI）

| 指標 | 目標値 | 測定方法 |
|------|--------|---------|
| **WFA ROBUST率** | >= 30% | Range期間のみで評価 |
| **平均PnL（ROBUST銘柄）** | >= +5% | OOS期間の複利PnL |
| **最大ドローダウン** | <= 20% | 全期間の最大DD |
| **トレード数（OOS）** | >= 30件 | 統計的有意性確保 |
| **勝率** | >= 70% | 小さい利幅のリピート戦略に適した水準 |

## 10. まとめ

トラリピはレンジ相場専用の戦略として、Prismの既存テンプレートに追加可能。

**実装アプローチ:**
1. **簡易版（オプションA）** で実現可能性を検証
2. **OOS/WFA** で銘柄横断の有効性を確認
3. 有望なら **完全版（オプションB）** で実運用レベルに拡張

**既存システムとの親和性:**
- `TrapGridCondition` を追加するだけで基本機能を実装可能
- Dual-TF EMA レジーム検出でトレンド期間を除外（リスク管理）
- パラメータ探索空間（128 configs）は過学習リスク低

**次のアクション:**
1. チームリーダーに設計承認を依頼
2. 承認後、フェーズ1（簡易版プロトタイプ）の実装を開始
