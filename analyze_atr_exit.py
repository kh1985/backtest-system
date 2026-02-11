#!/usr/bin/env python3
"""ATR動的exit WFA結果の詳細分析スクリプト"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_single_wfa(file_path: str, target_regime: str):
    """単一銘柄のWFA結果を分析"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    symbol = data['symbol']
    strategy = data['template_filter']  # adx_bb_long など

    # フォールド別詳細を集約
    fold_stats = []
    oos_pnls = []
    oos_trades_list = []

    for fold in data['folds']:
        selected = fold.get('selected', {}).get(target_regime, {})
        oos_result = fold.get('oos_results', {}).get(target_regime, {})

        if not selected or not oos_result:
            continue

        is_metrics = selected.get('metrics', {})
        oos_metrics = oos_result.get('metrics', {})

        is_pnl = is_metrics.get('total_pnl', 0.0)
        oos_pnl = oos_metrics.get('total_pnl', 0.0)

        fold_stats.append({
            'fold': fold['fold_index'],
            'is_pnl': is_pnl,
            'is_trades': is_metrics.get('trades', 0),
            'oos_pnl': oos_pnl,
            'oos_trades': oos_metrics.get('trades', 0),
            'oos_win_rate': oos_metrics.get('win_rate', 0.0),
            'oos_profit_factor': oos_metrics.get('profit_factor', 0.0),
            'oos_max_dd': oos_metrics.get('max_dd', 0.0),
        })

        oos_pnls.append(oos_pnl)
        oos_trades_list.append(oos_metrics.get('trades', 0))

    # CR計算（正のPnLフォールド比率）
    positive_folds = sum(1 for pnl in oos_pnls if pnl > 0)
    cr = positive_folds / len(oos_pnls) if oos_pnls else 0.0

    # 総合PnL（複利）
    total_pnl = 1.0
    for pnl in oos_pnls:
        total_pnl *= (1 + pnl / 100)
    total_pnl = (total_pnl - 1) * 100

    robust = cr >= 0.6

    return {
        'symbol': symbol,
        'strategy': strategy,
        'regime': target_regime,
        'cr': cr,
        'pnl': total_pnl,
        'robust': robust,
        'fold_stats': fold_stats,
        'avg_oos_trades': statistics.mean(oos_trades_list) if oos_trades_list else 0
    }

def analyze_batch_wfa(wfa_dir: Path, strategy_name: str, regime: str):
    """バッチWFA結果を分析"""
    results = []

    for wfa_file in sorted(wfa_dir.glob('*_wfa.json')):
        result = analyze_single_wfa(str(wfa_file), regime)
        results.append(result)

    # 全体統計
    total_symbols = len(results)
    robust_symbols = [r for r in results if r['robust']]
    robust_count = len(robust_symbols)
    robust_rate = robust_count / total_symbols * 100 if total_symbols > 0 else 0

    # ROBUST銘柄のPnL平均
    robust_pnls = [r['pnl'] for r in robust_symbols]
    avg_robust_pnl = statistics.mean(robust_pnls) if robust_pnls else 0.0

    # 最高PnL
    best_symbol = None
    best_pnl = -float('inf')
    for r in robust_symbols:
        if r['pnl'] > best_pnl:
            best_pnl = r['pnl']
            best_symbol = r['symbol']

    # 異常PnL（>1000%）の検出
    extreme_folds = []
    for result in results:
        for fold_stat in result['fold_stats']:
            if abs(fold_stat['oos_pnl']) > 1000:
                extreme_folds.append({
                    'symbol': result['symbol'],
                    'fold': fold_stat['fold'],
                    'oos_pnl': fold_stat['oos_pnl'],
                    'oos_trades': fold_stat['oos_trades'],
                    'oos_win_rate': fold_stat['oos_win_rate']
                })

    # フォールド別統計（全銘柄集計）
    all_fold_trades = defaultdict(list)
    all_fold_pnls = defaultdict(list)
    all_fold_win_rates = defaultdict(list)
    all_fold_pfs = defaultdict(list)
    all_fold_mdd = defaultdict(list)

    for result in results:
        for fold_stat in result['fold_stats']:
            fold_id = fold_stat['fold']
            all_fold_trades[fold_id].append(fold_stat['oos_trades'])
            all_fold_pnls[fold_id].append(fold_stat['oos_pnl'])
            all_fold_win_rates[fold_id].append(fold_stat['oos_win_rate'])
            all_fold_pfs[fold_id].append(fold_stat['oos_profit_factor'])
            all_fold_mdd[fold_id].append(fold_stat['oos_max_dd'])

    return {
        'strategy': strategy_name,
        'regime': regime,
        'total_symbols': total_symbols,
        'robust_count': robust_count,
        'robust_rate': robust_rate,
        'avg_robust_pnl': avg_robust_pnl,
        'best_symbol': best_symbol,
        'best_pnl': best_pnl,
        'results': results,
        'extreme_folds': extreme_folds,
        'fold_aggregates': {
            'trades': all_fold_trades,
            'pnls': all_fold_pnls,
            'win_rates': all_fold_win_rates,
            'profit_factors': all_fold_pfs,
            'max_dds': all_fold_mdd
        }
    }

def main():
    configs = [
        {
            'dir': '/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260211_153124_wfa/wfa',
            'name': 'adx_bb_long/uptrend/ATR(1.5/2.0)',
            'strategy': 'adx_bb_long',
            'regime': 'uptrend'
        },
        {
            'dir': '/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260211_153140_wfa/wfa',
            'name': 'ema_fast_cross_bb_short/downtrend/ATR(1.5/2.0)',
            'strategy': 'ema_fast_cross_bb_short',
            'regime': 'downtrend'
        }
    ]

    print("=" * 100)
    print("ATR動的exit WFA結果 詳細分析")
    print("=" * 100)
    print()

    all_analyses = []

    for config in configs:
        wfa_dir = Path(config['dir'])
        if not wfa_dir.exists():
            print(f"⚠️  ディレクトリが見つかりません: {wfa_dir}")
            continue

        print(f"## {config['name']}")
        print(f"ディレクトリ: {config['dir']}")
        print()

        analysis = analyze_batch_wfa(wfa_dir, config['strategy'], config['regime'])
        all_analyses.append({**analysis, **config})

        print(f"総銘柄数: {analysis['total_symbols']}")
        print(f"ROBUST数: {analysis['robust_count']} ({analysis['robust_rate']:.1f}%)")
        print(f"ROBUST銘柄平均PnL: {analysis['avg_robust_pnl']:+.1f}%")
        print(f"最高PnL銘柄: {analysis['best_symbol']} ({analysis['best_pnl']:+.1f}%)")
        print()

        # 銘柄別ROBUST詳細
        print("銘柄別詳細（CR>=0.6のみ）:")
        robust_results = sorted(
            [r for r in analysis['results'] if r['robust']],
            key=lambda x: x['pnl'],
            reverse=True
        )
        for r in robust_results:
            print(f"  {r['symbol']}: CR={r['cr']:.2f}, PnL={r['pnl']:+.1f}%")
        print()

        # フォールド別統計
        print("フォールド別集計（全銘柄平均）:")
        fold_agg = analysis['fold_aggregates']
        for fold_id in sorted(fold_agg['trades'].keys()):
            avg_trades = statistics.mean(fold_agg['trades'][fold_id]) if fold_agg['trades'][fold_id] else 0
            avg_pnl = statistics.mean(fold_agg['pnls'][fold_id]) if fold_agg['pnls'][fold_id] else 0
            avg_wr = statistics.mean(fold_agg['win_rates'][fold_id]) if fold_agg['win_rates'][fold_id] else 0
            pf_list = [x for x in fold_agg['profit_factors'][fold_id] if x > 0]
            avg_pf = statistics.mean(pf_list) if pf_list else 0
            mdd_list = [abs(x) for x in fold_agg['max_dds'][fold_id]]
            avg_mdd = statistics.mean(mdd_list) if mdd_list else 0
            print(f"  Fold {fold_id}: Trades={avg_trades:.1f}, PnL={avg_pnl:+.1f}%, WR={avg_wr:.1f}%, PF={avg_pf:.2f}, MaxDD={avg_mdd:.1f}%")
        print()

        # 異常PnL検出
        if analysis['extreme_folds']:
            print(f"⚠️  異常PnL検出（±1000%超）: {len(analysis['extreme_folds'])}件")
            for ex in analysis['extreme_folds'][:10]:
                print(f"  {ex['symbol']} Fold {ex['fold']}: PnL={ex['oos_pnl']:+.1f}%, Trades={ex['oos_trades']}, WR={ex['oos_win_rate']:.1f}%")
            print()
        else:
            print("異常PnL（±1000%超）: なし")
            print()

        print("-" * 100)
        print()

    # 比較サマリー
    print("=" * 100)
    print("ATR動的exit サマリー")
    print("=" * 100)
    print()
    print("| テンプレート | レジーム | Exit | ROBUST率 | 平均PnL | 最高銘柄 | 状態 |")
    print("|---|---|---|---|---|---|---|")
    for analysis in all_analyses:
        status = "✅" if analysis['robust_rate'] >= 25 and analysis['avg_robust_pnl'] > 0 else "⚠️"
        print(f"| {analysis['strategy']} | {analysis['regime']} | ATR(1.5/2.0) | "
              f"{analysis['robust_rate']:.0f}% ({analysis['robust_count']}/{analysis['total_symbols']}) | "
              f"{analysis['avg_robust_pnl']:+.1f}% | "
              f"{analysis['best_symbol']} ({analysis['best_pnl']:+.1f}%) | {status} |")
    print()

if __name__ == '__main__':
    main()
