# 検証ループ チーム編成ブループリント

## 基本構成: Modal/Local 2エージェント分離

```
[team-lead] リーダー（自分）
  ├── [local-analyzer] 分析・判断・テンプレート修正（ローカル専門）
  └── [modal-runner]   Modal実行・DL（クラウド専門）
```

### local-analyzer（頭脳）
- **担当**: JSON分析、PASS率計算、過学習トラップ特定、テンプレート修正、判断
- **ツール**: Bash(python), Read, Edit, Grep, Glob
- **Modalコマンド実行禁止**
- 判断後にmodal-runnerへ次のコマンドを送信

### modal-runner（実行者）
- **担当**: `python3 -m modal run` の実行のみ
- **ツール**: Bash
- **分析・テンプレート修正禁止**
- local-analyzerからの指示待ち → 実行 → 完了報告

## 判断ルール（local-analyzerに埋め込む）

```
1. 閾値別PASS率を計算
2. PASS率 0% の閾値 → 過学習トラップとして除外
3. 上位2-3値に絞り込み → テンプレート修正 → modal-runnerに再実行指示
4. 最良値を固定（1値）→ 最終テスト指示
5. 50%+ かつ n≥10 達成 → 完了報告
6. 未達 → その旨報告（これ以上の改善は困難）
```

## 拡張パターン

### A. 独立戦略の並列検証
```
[team-lead]
  ├── [analyzer-long]  ロング系分析
  ├── [analyzer-short] ショート系分析
  └── [modal-runner]   共有Modal実行（指示元を区別）
```

### B. 複数タスク並列（Step 11型）
```
[team-lead]
  ├── [yaml-exporter]  YAMLエクスポート（ローカル完結）
  ├── [wfa-runner]     WFA検証（ローカル or Modal）
  └── [dt-explorer]    新テンプレート探索（Modal + 分析）
      ├── 自分で分析もする（1人で完結）
      └── または modal-runner に指示する
```

## Modal実行コマンドテンプレート

### 最適化
```bash
python3 -m modal run scripts/modal_optimize.py \
  --period "20230201-20240131,20240201-20250131,20250201-20260130" \
  --exit-profiles atr_compact \
  --templates "{テンプレート名}" \
  --super-htf 4h
```

### 結果DL
```bash
python3 -m modal run scripts/modal_download.py --run-id {RUN_ID}
```

## 分析スクリプトテンプレート

```python
import json, glob, os
from collections import defaultdict

BASE = 'results/batch/{RUN_ID}/optimization/'
MIN_TRADES = 20

records = []
for fpath in sorted(glob.glob(os.path.join(BASE, '*.json'))):
    with open(fpath) as f:
        data = json.load(f)
    symbol = data['symbol']
    period = data['period']
    for regime, result in data.get('test_results', {}).items():
        metrics = result.get('metrics', {})
        template = result.get('template', '')
        base_template = template.rsplit('_atr_', 1)[0] if '_atr_' in template else template
        exit_profile = template.split('_atr_')[-1] if '_atr_' in template else ''
        params = result.get('params', {})
        records.append({
            'symbol': symbol, 'period': period, 'regime': regime,
            'template': base_template, 'exit': exit_profile,
            'params': params,
            'trades': metrics.get('trades', 0),
            'pnl': metrics.get('total_pnl', 0),
            'win_rate': metrics.get('win_rate', 0),
        })

def is_pass(r):
    return r['pnl'] > 0 and r['trades'] >= MIN_TRADES

# レジーム別
for regime in ['uptrend', 'downtrend', 'range']:
    subset = [r for r in records if r['regime'] == regime]
    n = len(subset)
    p = sum(1 for r in subset if is_pass(r))
    rate = p/n*100 if n > 0 else 0
    print(f'{regime:<12} {p}/{n} ({rate:.1f}%)')

# パラメータ別（例: RSI閾値）
param_name = 'rsi_threshold'  # 変更可能
vals = sorted(set(r['params'].get(param_name) for r in records if param_name in r['params']))
for val in vals:
    subset = [r for r in records if r['params'].get(param_name) == val]
    n = len(subset)
    p = sum(1 for r in subset if is_pass(r))
    rate = p/n*100 if n > 0 else 0
    trap = ' ← 過学習トラップ' if rate == 0 and n >= 5 else ''
    print(f'  {param_name}={val}: {p}/{n} ({rate:.1f}%){trap}')
```

## 完了基準
- 全タスクが completed
- ロードマップ・MEMORY.md・セッションログ更新済み
- team-lead がチーム削除（TeamDelete）
