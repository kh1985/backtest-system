#!/usr/bin/env python3
"""固定exit vs ATR exit の銘柄別詳細比較"""

import json
from pathlib import Path
from collections import defaultdict

def load_fixed_exit_results(wfa_file: str, template: str, regime: str):
    """固定exit WFA結果から銘柄別データを取得"""
    with open(wfa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    symbol_results = {}
    for entry in data['results']:
        if entry['template'] == template and regime in entry.get('regimes', {}):
            symbol = entry['symbol']
            period = entry['period']
            regime_data = entry['regimes'][regime]

            if symbol not in symbol_results:
                symbol_results[symbol] = []

            symbol_results[symbol].append({
                'period': period,
                'cr': regime_data.get('consistency_ratio', 0.0),
                'pnl': regime_data.get('stitched_oos_pnl', 0.0),
                'robust': regime_data.get('robust', False)
            })

    # 3期間平均を計算
    symbol_summary = {}
    for symbol, periods in symbol_results.items():
        robust_count = sum(1 for p in periods if p['cr'] >= 0.6)
        avg_cr = sum(p['cr'] for p in periods) / len(periods)
        avg_pnl = sum(p['pnl'] for p in periods) / len(periods)

        symbol_summary[symbol] = {
            'robust_count': robust_count,
            'total_periods': len(periods),
            'avg_cr': avg_cr,
            'avg_pnl': avg_pnl,
            'robust': robust_count >= 2  # 3期間中2期間以上ROBUST
        }

    return symbol_summary

def load_atr_exit_results(wfa_dir: Path, regime: str):
    """ATR exit WFA結果から銘柄別データを取得"""
    symbol_results = {}

    for wfa_file in wfa_dir.glob('*_wfa.json'):
        with open(wfa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        symbol = data['symbol']
        oos_pnls = []

        for fold in data['folds']:
            oos_result = fold.get('oos_results', {}).get(regime, {})
            if oos_result:
                oos_metrics = oos_result.get('metrics', {})
                oos_pnls.append(oos_metrics.get('total_pnl', 0.0))

        if oos_pnls:
            positive_folds = sum(1 for pnl in oos_pnls if pnl > 0)
            cr = positive_folds / len(oos_pnls)

            # 複利計算
            total_pnl = 1.0
            for pnl in oos_pnls:
                total_pnl *= (1 + pnl / 100)
            total_pnl = (total_pnl - 1) * 100

            symbol_results[symbol] = {
                'cr': cr,
                'pnl': total_pnl,
                'robust': cr >= 0.6,
                'n_folds': len(oos_pnls)
            }

    return symbol_results

def main():
    print("=" * 120)
    print("固定exit vs ATR exit 銘柄別詳細比較")
    print("=" * 120)
    print()

    # adx_bb_long の比較
    print("## 1. adx_bb_long (Uptrend)")
    print()

    fixed_results = load_fixed_exit_results(
        '/Users/kenjihachiya/Desktop/work/development/backtest-system/results/wfa/wfa_20260211_095148.json',
        'adx_bb_long',
        'uptrend'
    )

    atr_results = load_atr_exit_results(
        Path('/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260211_153124_wfa/wfa'),
        'uptrend'
    )

    # 共通銘柄のみ比較
    common_symbols = set(fixed_results.keys()) & set(atr_results.keys())

    print(f"比較対象銘柄数: {len(common_symbols)}")
    print()
    print("| 銘柄 | 固定tp30 CR | 固定PnL | ATR CR | ATR PnL | 優位性 | 改善率 |")
    print("|---|---|---|---|---|---|---|")

    comparison_data = []
    for symbol in sorted(common_symbols):
        fixed = fixed_results[symbol]
        atr = atr_results[symbol]

        fixed_cr = fixed['avg_cr']
        fixed_pnl = fixed['avg_pnl']
        atr_cr = atr['cr']
        atr_pnl = atr['pnl']

        improvement = ((atr_pnl - fixed_pnl) / abs(fixed_pnl) * 100) if fixed_pnl != 0 else 0

        if atr['robust'] and not fixed['robust']:
            advantage = "🟢 ATR単独"
        elif not atr['robust'] and fixed['robust']:
            advantage = "🔴 固定単独"
        elif atr['robust'] and fixed['robust']:
            if atr_pnl > fixed_pnl * 1.5:
                advantage = "🟢 ATR大優位"
            elif atr_pnl > fixed_pnl:
                advantage = "🟢 ATR優位"
            elif fixed_pnl > atr_pnl * 1.5:
                advantage = "🔴 固定大優位"
            elif fixed_pnl > atr_pnl:
                advantage = "🔴 固定優位"
            else:
                advantage = "⚪ 同等"
        else:
            advantage = "⚫ 両方NG"

        comparison_data.append({
            'symbol': symbol,
            'fixed_cr': fixed_cr,
            'fixed_pnl': fixed_pnl,
            'atr_cr': atr_cr,
            'atr_pnl': atr_pnl,
            'advantage': advantage,
            'improvement': improvement
        })

        print(f"| {symbol} | {fixed_cr:.2f} | {fixed_pnl:+.1f}% | {atr_cr:.2f} | {atr_pnl:+.1f}% | {advantage} | {improvement:+.0f}% |")

    print()
    print("凡例:")
    print("  🟢 ATR大優位: ATR PnL > 固定 PnL × 1.5")
    print("  🟢 ATR優位: ATR PnL > 固定 PnL")
    print("  🔴 固定優位: 固定 PnL > ATR PnL")
    print("  🟢 ATR単独: ATRのみROBUST（CR>=0.6）")
    print("  🔴 固定単独: 固定のみROBUST")
    print("  ⚪ 同等: 両方ROBUSTでPnL差<50%")
    print("  ⚫ 両方NG: 両方ともROBUST未達")
    print()

    # 統計サマリー
    atr_wins = sum(1 for d in comparison_data if '🟢' in d['advantage'])
    fixed_wins = sum(1 for d in comparison_data if '🔴' in d['advantage'])
    ties = sum(1 for d in comparison_data if '⚪' in d['advantage'])
    both_ng = sum(1 for d in comparison_data if '⚫' in d['advantage'])

    print("統計サマリー:")
    print(f"  ATR優位: {atr_wins}銘柄 ({atr_wins/len(common_symbols)*100:.1f}%)")
    print(f"  固定優位: {fixed_wins}銘柄 ({fixed_wins/len(common_symbols)*100:.1f}%)")
    print(f"  同等: {ties}銘柄")
    print(f"  両方NG: {both_ng}銘柄")
    print()

    # 平均改善率
    robust_improvements = [d['improvement'] for d in comparison_data if '🟢' in d['advantage']]
    if robust_improvements:
        avg_improvement = sum(robust_improvements) / len(robust_improvements)
        print(f"ATR優位銘柄の平均改善率: {avg_improvement:+.1f}%")
        print()

    print("-" * 120)
    print()

    # ema_fast_cross_bb_short の比較
    print("## 2. ema_fast_cross_bb_short (Downtrend)")
    print()

    fixed_results_dt = load_fixed_exit_results(
        '/Users/kenjihachiya/Desktop/work/development/backtest-system/results/wfa/wfa_20260211_095225.json',
        'ema_fast_cross_bb_short',
        'downtrend'
    )

    atr_results_dt = load_atr_exit_results(
        Path('/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260211_153140_wfa/wfa'),
        'downtrend'
    )

    common_symbols_dt = set(fixed_results_dt.keys()) & set(atr_results_dt.keys())

    print(f"比較対象銘柄数: {len(common_symbols_dt)}")
    print()
    print("| 銘柄 | 固定tp15 CR | 固定PnL | ATR CR | ATR PnL | 優位性 | 改善率 |")
    print("|---|---|---|---|---|---|---|")

    comparison_data_dt = []
    for symbol in sorted(common_symbols_dt):
        fixed = fixed_results_dt[symbol]
        atr = atr_results_dt[symbol]

        fixed_cr = fixed['avg_cr']
        fixed_pnl = fixed['avg_pnl']
        atr_cr = atr['cr']
        atr_pnl = atr['pnl']

        improvement = ((atr_pnl - fixed_pnl) / abs(fixed_pnl) * 100) if fixed_pnl != 0 else 0

        if atr['robust'] and not fixed['robust']:
            advantage = "🟢 ATR単独"
        elif not atr['robust'] and fixed['robust']:
            advantage = "🔴 固定単独"
        elif atr['robust'] and fixed['robust']:
            if atr_pnl > fixed_pnl * 1.5:
                advantage = "🟢 ATR大優位"
            elif atr_pnl > fixed_pnl:
                advantage = "🟢 ATR優位"
            elif fixed_pnl > atr_pnl * 1.5:
                advantage = "🔴 固定大優位"
            elif fixed_pnl > atr_pnl:
                advantage = "🔴 固定優位"
            else:
                advantage = "⚪ 同等"
        else:
            advantage = "⚫ 両方NG"

        comparison_data_dt.append({
            'symbol': symbol,
            'fixed_cr': fixed_cr,
            'fixed_pnl': fixed_pnl,
            'atr_cr': atr_cr,
            'atr_pnl': atr_pnl,
            'advantage': advantage,
            'improvement': improvement
        })

        print(f"| {symbol} | {fixed_cr:.2f} | {fixed_pnl:+.1f}% | {atr_cr:.2f} | {atr_pnl:+.1f}% | {advantage} | {improvement:+.0f}% |")

    print()

    # 統計サマリー
    atr_wins_dt = sum(1 for d in comparison_data_dt if '🟢' in d['advantage'])
    fixed_wins_dt = sum(1 for d in comparison_data_dt if '🔴' in d['advantage'])
    ties_dt = sum(1 for d in comparison_data_dt if '⚪' in d['advantage'])
    both_ng_dt = sum(1 for d in comparison_data_dt if '⚫' in d['advantage'])

    print("統計サマリー:")
    print(f"  ATR優位: {atr_wins_dt}銘柄 ({atr_wins_dt/len(common_symbols_dt)*100:.1f}%)")
    print(f"  固定優位: {fixed_wins_dt}銘柄 ({fixed_wins_dt/len(common_symbols_dt)*100:.1f}%)")
    print(f"  同等: {ties_dt}銘柄")
    print(f"  両方NG: {both_ng_dt}銘柄")
    print()

if __name__ == '__main__':
    main()
