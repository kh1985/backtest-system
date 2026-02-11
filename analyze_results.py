#!/usr/bin/env python3
"""
85個のJSON結果ファイルを分析して詳細レポートを生成
"""
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# JSONファイルのディレクトリ
RESULTS_DIR = Path("/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260206_200522/optimization")

def load_all_results():
    """全JSONファイルを読み込み"""
    results = []
    json_files = sorted(RESULTS_DIR.glob("*.json"))

    for filepath in json_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)

    return results

def extract_oos_results(data: dict) -> Dict[str, Dict]:
    """
    test_resultsから各regimeのOOS結果を抽出
    """
    oos_results = {}
    test_results = data.get('test_results', {})

    for regime in ['uptrend', 'downtrend', 'range']:
        if regime in test_results:
            result = test_results[regime]
            metrics = result.get('metrics', {})

            # OOS PASS判定: pnl > 0 かつ profit_factor >= 1.0
            pnl = metrics.get('total_pnl', 0)
            pf = metrics.get('profit_factor', 0)
            oos_pass = pnl > 0 and pf >= 1.0

            oos_results[regime] = {
                'template': result.get('template', ''),
                'exit_profile': result.get('exit_profile', ''),
                'pnl_pct': pnl,
                'total_trades': metrics.get('trades', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': pf,
                'sharpe_ratio': metrics.get('sharpe', 0),
                'oos_pass': oos_pass
            }

    return oos_results

def analyze_all():
    """全体分析を実行"""
    print("Loading all result files...")
    all_results = load_all_results()
    print(f"Loaded {len(all_results)} files")

    # データ構造: symbol -> period -> regime -> result
    data_by_symbol = defaultdict(lambda: defaultdict(dict))

    # テンプレート別の集計
    template_stats = defaultdict(lambda: {'pass': 0, 'fail': 0})

    # Exit profile別の集計
    exit_stats = defaultdict(lambda: {'pass': 0, 'fail': 0, 'total_pnl': 0})

    # レジーム別の集計
    regime_stats = defaultdict(lambda: {'pass': 0, 'fail': 0, 'symbols_pass': set()})

    for data in all_results:
        symbol = data['symbol']
        period = data['period']

        oos_results = extract_oos_results(data)

        for regime, result in oos_results.items():
            data_by_symbol[symbol][period][regime] = result

            template = result['template']
            exit_profile = result['exit_profile']
            oos_pass = result['oos_pass']

            # テンプレート統計
            if oos_pass:
                template_stats[template]['pass'] += 1
            else:
                template_stats[template]['fail'] += 1

            # Exit profile統計
            if oos_pass:
                exit_stats[exit_profile]['pass'] += 1
                exit_stats[exit_profile]['total_pnl'] += result['pnl_pct']
            else:
                exit_stats[exit_profile]['fail'] += 1

            # レジーム統計
            if oos_pass:
                regime_stats[regime]['pass'] += 1
                regime_stats[regime]['symbols_pass'].add(symbol)
            else:
                regime_stats[regime]['fail'] += 1

    # マークダウンレポート生成
    print("\n" + "="*80)
    print("# 85銘柄 × 3期間 バックテスト結果分析")
    print("="*80 + "\n")

    # 1. rsi_bb_reversal_short のみ: 銘柄×レジーム別 OOS PASS一覧
    print("## 1. 銘柄×レジーム別 OOS PASS一覧（rsi_bb_reversal_short のみ）\n")

    # 期間別にグループ化
    periods = ['20230201-20240131', '20240201-20250131', '20250201-20260130']
    period_labels = ['2023-2024', '2024-2025', '2025-2026']

    for period, label in zip(periods, period_labels):
        print(f"### {label}\n")
        print("| Symbol | Uptrend | Downtrend | Range |")
        print("|--------|---------|-----------|-------|")

        # 銘柄をアルファベット順にソート
        sorted_symbols = sorted([s for s in data_by_symbol.keys() if period in data_by_symbol[s]])

        for symbol in sorted_symbols:
            if period not in data_by_symbol[symbol]:
                continue

            row = [symbol]
            for regime in ['uptrend', 'downtrend', 'range']:
                result = data_by_symbol[symbol][period].get(regime)
                if result and result['template'] == 'rsi_bb_reversal_short':
                    if result['oos_pass']:
                        cell = f"✅ {result['pnl_pct']:.1f}%"
                    else:
                        cell = f"❌ {result['pnl_pct']:.1f}%"
                else:
                    cell = "-"
                row.append(cell)

            print("| " + " | ".join(row) + " |")

        print()

    # 2. レジーム別の詳細統計
    print("## 2. レジーム別の詳細統計\n")

    for regime in ['uptrend', 'downtrend', 'range']:
        stats = regime_stats[regime]
        total = stats['pass'] + stats['fail']
        pass_rate = 100 * stats['pass'] / total if total > 0 else 0

        print(f"### {regime.capitalize()}\n")
        print(f"- **OOS PASS率**: {stats['pass']}/{total} ({pass_rate:.1f}%)")
        print(f"- **通過銘柄数**: {len(stats['symbols_pass'])} 銘柄")

        # 通過銘柄をリストアップ
        passing_symbols = []
        for symbol in sorted(stats['symbols_pass']):
            for period in periods:
                if period in data_by_symbol[symbol]:
                    result = data_by_symbol[symbol][period].get(regime)
                    if result and result['oos_pass']:
                        passing_symbols.append(f"{symbol} ({period[:4]}年)")

        if passing_symbols:
            print(f"- **通過銘柄**: {', '.join(passing_symbols[:10])}" +
                  (f"... 他{len(passing_symbols)-10}件" if len(passing_symbols) > 10 else ""))

        print()

    # 3. Exit profile比較
    print("## 3. Exit Profile比較\n")
    print("| Exit Profile | PASS | FAIL | PASS率 | 平均PnL(PASS) |")
    print("|--------------|------|------|--------|---------------|")

    for profile in sorted(exit_stats.keys()):
        stats = exit_stats[profile]
        total = stats['pass'] + stats['fail']
        pass_rate = 100 * stats['pass'] / total if total > 0 else 0
        avg_pnl = stats['total_pnl'] / stats['pass'] if stats['pass'] > 0 else 0

        print(f"| {profile} | {stats['pass']} | {stats['fail']} | {pass_rate:.1f}% | {avg_pnl:.2f}% |")

    print()

    # 4. rsi_bb_reversal_short vs bb_volume_reversal_short
    print("## 4. テンプレート比較\n")
    print("| Template | PASS | FAIL | PASS率 |")
    print("|----------|------|------|--------|")

    for template in sorted(template_stats.keys()):
        stats = template_stats[template]
        total = stats['pass'] + stats['fail']
        pass_rate = 100 * stats['pass'] / total if total > 0 else 0

        print(f"| {template} | {stats['pass']} | {stats['fail']} | {pass_rate:.1f}% |")

    print()

    # 5. 期間一貫性分析
    print("## 5. 期間一貫性（複数期間でPASSした銘柄）\n")

    for regime in ['uptrend', 'downtrend', 'range']:
        print(f"### {regime.capitalize()}\n")

        # 各銘柄が何期間PASSしたかカウント
        symbol_pass_count = defaultdict(list)

        for symbol in data_by_symbol:
            for period in periods:
                if period in data_by_symbol[symbol]:
                    result = data_by_symbol[symbol][period].get(regime)
                    if result and result['oos_pass']:
                        symbol_pass_count[symbol].append(period)

        # 2期間以上PASSした銘柄を表示
        multi_pass = {s: periods for s, periods in symbol_pass_count.items() if len(periods) >= 2}

        if multi_pass:
            print(f"**2期間以上PASS: {len(multi_pass)}銘柄**\n")
            for symbol in sorted(multi_pass.keys()):
                period_str = ', '.join([p[:4] for p in multi_pass[symbol]])
                print(f"- {symbol}: {period_str}年")
        else:
            print("2期間以上PASSした銘柄なし")

        print()

if __name__ == '__main__':
    analyze_all()
