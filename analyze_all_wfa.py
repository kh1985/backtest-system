"""
全WFA結果を集計して、実運用候補を抽出

バグ修正後（2026-02-11）の全WFA結果から：
- ROBUST率（CR>=0.6の銘柄数/全銘柄数）
- 平均PnL
を集計し、レジーム別に整理する。

実運用基準:
- ROBUST率 >= 30%
- 平均PnL > 0%
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_wfa_results(results_dir: Path) -> List[Dict]:
    """2026-02-11のWFA結果ファイルを全て読み込む"""
    pattern = "wfa_20260211_*.json"
    files = sorted(results_dir.glob(pattern))

    results = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append({
                'file': file_path.name,
                'run_id': data.get('run_id'),
                'strategies': data.get('strategies', []),
                'summary': data.get('summary', {})
            })

    return results

def extract_strategy_regime_data(all_results: List[Dict]) -> Dict[str, Dict]:
    """戦略×レジーム別のデータを抽出"""
    aggregated = {}

    for result in all_results:
        summary = result['summary']
        for key, data in summary.items():
            # key format: "template_name/regime" or "template_name"
            if key not in aggregated:
                aggregated[key] = {
                    'files': [],
                    'robust': data.get('robust', 0),
                    'total': data.get('total', 0),
                    'avg_pnl': data.get('avg_pnl', 0),
                    'avg_cr': data.get('avg_cr', 0),
                    'avg_wfe': data.get('avg_wfe', 0)
                }
            aggregated[key]['files'].append(result['file'])

    return aggregated

def categorize_by_regime(data: Dict[str, Dict]) -> Dict[str, List[Tuple[str, Dict]]]:
    """レジーム別に分類"""
    categorized = {
        'uptrend': [],
        'downtrend': [],
        'range': [],
        'unknown': []
    }

    for key, values in data.items():
        if '/uptrend' in key:
            regime = 'uptrend'
            template = key.replace('/uptrend', '')
        elif '/downtrend' in key:
            regime = 'downtrend'
            template = key.replace('/downtrend', '')
        elif '/range' in key:
            regime = 'range'
            template = key.replace('/range', '')
        else:
            regime = 'unknown'
            template = key

        robust_rate = 100 * values['robust'] / max(values['total'], 1)
        categorized[regime].append((template, {
            **values,
            'robust_rate': robust_rate
        }))

    # ROBUST率でソート
    for regime in categorized:
        categorized[regime].sort(key=lambda x: x[1]['robust_rate'], reverse=True)

    return categorized

def print_results(categorized: Dict[str, List[Tuple[str, Dict]]]):
    """結果を整形して出力"""
    print("=" * 100)
    print("  バグ修正後（2026-02-11）全WFA結果集計")
    print("=" * 100)
    print()

    # 実運用基準
    ROBUST_THRESHOLD = 30.0
    PNL_THRESHOLD = 0.0

    for regime in ['uptrend', 'downtrend', 'range']:
        strategies = categorized[regime]
        if not strategies:
            continue

        regime_label = {
            'uptrend': '📈 UPTREND（ロング戦略）',
            'downtrend': '📉 DOWNTREND（ショート戦略）',
            'range': '↔️  RANGE（レンジ戦略）'
        }[regime]

        print(f"\n{regime_label}")
        print("-" * 100)
        print(f"{'戦略':<40} {'ROBUST率':<15} {'平均PnL':<12} {'平均CR':<10} {'判定':<10}")
        print("-" * 100)

        for template, data in strategies:
            robust_str = f"{data['robust']}/{data['total']} ({data['robust_rate']:.1f}%)"
            pnl_str = f"{data['avg_pnl']:+.1f}%"
            cr_str = f"{data['avg_cr']:.3f}"

            # 判定
            is_robust = data['robust_rate'] >= ROBUST_THRESHOLD
            is_profitable = data['avg_pnl'] > PNL_THRESHOLD

            if is_robust and is_profitable:
                status = "✅ 実運用候補"
            elif is_robust:
                status = "⚠️ 要注意"
            else:
                status = "❌ 除外"

            print(f"{template:<40} {robust_str:<15} {pnl_str:<12} {cr_str:<10} {status:<10}")

    print("\n" + "=" * 100)
    print("  実運用推奨戦略（ROBUST率>=30% & 平均PnL>0%）")
    print("=" * 100)
    print()

    candidates = []
    for regime in ['uptrend', 'downtrend', 'range']:
        for template, data in categorized[regime]:
            if data['robust_rate'] >= ROBUST_THRESHOLD and data['avg_pnl'] > PNL_THRESHOLD:
                candidates.append({
                    'regime': regime,
                    'template': template,
                    'robust_rate': data['robust_rate'],
                    'robust': data['robust'],
                    'total': data['total'],
                    'avg_pnl': data['avg_pnl'],
                    'avg_cr': data['avg_cr']
                })

    if candidates:
        print(f"{'レジーム':<15} {'戦略':<40} {'ROBUST率':<15} {'平均PnL':<12}")
        print("-" * 90)
        for c in sorted(candidates, key=lambda x: x['robust_rate'], reverse=True):
            robust_str = f"{c['robust']}/{c['total']} ({c['robust_rate']:.1f}%)"
            pnl_str = f"{c['avg_pnl']:+.1f}%"
            print(f"{c['regime']:<15} {c['template']:<40} {robust_str:<15} {pnl_str:<12}")
    else:
        print("⚠️ 実運用基準を満たす戦略がありません。")

    print()

def main():
    results_dir = Path(__file__).resolve().parent / "results" / "wfa"

    print("WFA結果ファイルを読み込み中...")
    all_results = load_wfa_results(results_dir)
    print(f"  読み込み完了: {len(all_results)}件\n")

    print("戦略×レジーム別に集計中...")
    aggregated = extract_strategy_regime_data(all_results)

    print("レジーム別に分類中...")
    categorized = categorize_by_regime(aggregated)

    print_results(categorized)

if __name__ == "__main__":
    main()
