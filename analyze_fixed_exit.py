#!/usr/bin/env python3
"""固定exit WFA結果の集計スクリプト"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_wfa_result(file_path: str, strategy_name: str, regime: str):
    """WFA結果ファイルを分析"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # resultsから該当戦略・レジームの結果を抽出
    symbol_results = defaultdict(list)

    for entry in data['results']:
        if entry['template'] == strategy_name and regime in entry.get('regimes', {}):
            symbol = entry['symbol']
            period = entry['period']
            regime_data = entry['regimes'][regime]

            symbol_results[symbol].append({
                'period': period,
                'cr': regime_data.get('consistency_ratio', 0.0),
                'pnl': regime_data.get('stitched_oos_pnl', 0.0),
                'robust': regime_data.get('robust', False),
                'wfe': regime_data.get('wfe', 0.0),
                'n_folds': entry.get('n_folds', 5)
            })

    # 銘柄ごとのROBUST判定（3期間中何期間がROBUST）
    symbol_stats = {}
    for symbol, periods in symbol_results.items():
        robust_count = sum(1 for p in periods if p['cr'] >= 0.6)
        total_periods = len(periods)
        avg_pnl = sum(p['pnl'] for p in periods) / total_periods if total_periods > 0 else 0.0

        symbol_stats[symbol] = {
            'robust_count': robust_count,
            'total_periods': total_periods,
            'avg_pnl': avg_pnl,
            'periods': periods
        }

    # 全体統計（CR>=0.6の銘柄×期間を集計）
    total_tests = sum(s['total_periods'] for s in symbol_stats.values())
    robust_tests = sum(s['robust_count'] for s in symbol_stats.values())
    robust_rate = robust_tests / total_tests * 100 if total_tests > 0 else 0

    # ROBUST銘柄のPnL平均（少なくとも1期間がROBUST）
    robust_symbols = [s for s, stat in symbol_stats.items() if stat['robust_count'] > 0]
    robust_pnls = [symbol_stats[s]['avg_pnl'] for s in robust_symbols]
    avg_robust_pnl = sum(robust_pnls) / len(robust_pnls) if robust_pnls else 0.0

    # 最高PnL銘柄
    best_symbol = None
    best_pnl = -float('inf')
    for symbol in robust_symbols:
        pnl = symbol_stats[symbol]['avg_pnl']
        if pnl > best_pnl:
            best_pnl = pnl
            best_symbol = symbol

    return {
        'total_tests': total_tests,
        'robust_tests': robust_tests,
        'robust_rate': robust_rate,
        'avg_robust_pnl': avg_robust_pnl,
        'symbol_stats': symbol_stats,
        'robust_symbols': robust_symbols,
        'best_symbol': best_symbol,
        'best_pnl': best_pnl
    }

def main():
    base_dir = Path('/Users/kenjihachiya/Desktop/work/development/backtest-system/results/wfa')

    configs = [
        {
            'file': 'wfa_20260211_094908.json',
            'name': 'rsi_bb_long_f35/tp20',
            'template': 'rsi_bb_long_f35',
            'regime': 'uptrend',
            'exit': 'tp20固定'
        },
        {
            'file': 'wfa_20260211_095121.json',
            'name': 'rsi_bb_long_f35/tp30',
            'template': 'rsi_bb_long_f35',
            'regime': 'uptrend',
            'exit': 'tp30固定'
        },
        {
            'file': 'wfa_20260211_095148.json',
            'name': 'adx_bb_long/tp30',
            'template': 'adx_bb_long',
            'regime': 'uptrend',
            'exit': 'tp30固定'
        },
        {
            'file': 'wfa_20260211_095225.json',
            'name': 'ema_fast_cross_bb_short/tp15',
            'template': 'ema_fast_cross_bb_short',
            'regime': 'downtrend',
            'exit': 'tp15固定'
        }
    ]

    print("=" * 100)
    print("固定exit WFA結果集計（CR>=0.6基準）")
    print("=" * 100)
    print()

    results = []

    for config in configs:
        file_path = base_dir / config['file']
        if not file_path.exists():
            print(f"⚠️  ファイルが見つかりません: {file_path}")
            continue

        print(f"## {config['name']}")
        print(f"ファイル: {config['file']}")
        print()

        result = analyze_wfa_result(str(file_path), config['template'], config['regime'])
        result.update(config)
        results.append(result)

        print(f"総テスト数: {result['total_tests']} (銘柄×期間)")
        print(f"ROBUST数: {result['robust_tests']} ({result['robust_rate']:.1f}%)")
        print(f"ROBUST銘柄の平均PnL: {result['avg_robust_pnl']:+.1f}%")
        print(f"最高PnL銘柄: {result['best_symbol']} ({result['best_pnl']:+.1f}%)")
        print()

        # 銘柄別詳細（ROBUST回数降順）
        print("銘柄別ROBUST詳細（1期間以上ROBUSTのみ）:")
        sorted_symbols = sorted(
            result['robust_symbols'],
            key=lambda s: (result['symbol_stats'][s]['robust_count'], result['symbol_stats'][s]['avg_pnl']),
            reverse=True
        )
        for symbol in sorted_symbols[:15]:
            stat = result['symbol_stats'][symbol]
            print(f"  {symbol}: {stat['robust_count']}/{stat['total_periods']} ROBUST, 平均PnL {stat['avg_pnl']:+.1f}%")

        print()
        print("-" * 100)
        print()

    # サマリーテーブル
    print("=" * 100)
    print("サマリーテーブル")
    print("=" * 100)
    print()
    print("| テンプレート | レジーム | Exit | ROBUST率 | 平均PnL | 最高銘柄 | 状態 |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        status = "✅" if r['robust_rate'] >= 25 and r['avg_robust_pnl'] > 0 else "⚠️"
        print(f"| {r['template']} | {r['regime']} | {r['exit']} | "
              f"{r['robust_rate']:.0f}% ({r['robust_tests']}/{r['total_tests']}) | "
              f"{r['avg_robust_pnl']:+.1f}% | "
              f"{r['best_symbol']} ({r['best_pnl']:+.1f}%) | {status} |")
    print()

if __name__ == '__main__':
    main()
