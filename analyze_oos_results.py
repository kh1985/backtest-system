#!/usr/bin/env python3
"""
OOS結果分析スクリプト
全85ファイルを読み込み、rsi_bb_reversal_short × atr_tp30_sl20 と bb_volume_reversal_short × atr_tp30_sl20 の
銘柄×期間×レジーム詳細テーブルを生成
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ディレクトリパス
RESULTS_DIR = Path("/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260206_200522/optimization")

def parse_filename(filename: str) -> Tuple[str, str]:
    """ファイル名から銘柄と期間を抽出"""
    # 例: AAVEUSDT_20230201-20240131_15m_1h.json
    parts = filename.replace('.json', '').split('_')
    symbol = parts[0]
    period = parts[1]
    return symbol, period

def get_oos_result(test_results: dict, regime: str, template: str, exit_profile: str) -> dict:
    """
    test_resultsから指定されたregimeのOOS結果を取得
    指定されたtemplate & exit_profileとマッチするかチェック
    """
    if regime not in test_results:
        return None

    result = test_results[regime]

    # テンプレートとexit_profileがマッチするかチェック
    if result.get('template') == template and result.get('exit_profile') == exit_profile:
        return result

    return None

def format_cell(result: dict) -> str:
    """セルのフォーマット: PASS +X.X% (Ntrades) or FAIL -X.X% (Ntrades) or N/A"""
    if result is None:
        return "N/A"

    metrics = result.get('metrics', {})
    pnl = metrics.get('total_pnl', 0)
    trades = metrics.get('trades', 0)

    if pnl > 0:
        return f"PASS +{pnl:.1f}% ({trades})"
    else:
        return f"FAIL {pnl:.1f}% ({trades})"

def main():
    # データ収集
    data = {}  # {(symbol, period): {regime: result}}

    # 全ファイル読み込み
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith('.json'):
            continue

        filepath = RESULTS_DIR / filename
        symbol, period = parse_filename(filename)

        with open(filepath, 'r') as f:
            content = json.load(f)

        test_results = content.get('test_results', {})

        # データ格納
        key = (symbol, period)
        data[key] = test_results

    # TABLE 1: rsi_bb_reversal_short × atr_tp30_sl20
    print("### TABLE 1: rsi_bb_reversal_short × atr_tp30_sl20 の銘柄×期間×レジーム詳細\n")
    print("| 銘柄 | 期間 | Uptrend | Downtrend | Range |")
    print("|------|------|---------|-----------|-------|")

    template1 = "rsi_bb_reversal_short"
    exit1 = "atr_tp30_sl20"

    # 銘柄名でソート
    for (symbol, period), test_results in sorted(data.items()):
        uptrend = get_oos_result(test_results, 'uptrend', template1, exit1)
        downtrend = get_oos_result(test_results, 'downtrend', template1, exit1)
        range_regime = get_oos_result(test_results, 'range', template1, exit1)

        print(f"| {symbol} | {period} | {format_cell(uptrend)} | {format_cell(downtrend)} | {format_cell(range_regime)} |")

    # TABLE 2: 期間一貫性
    print("\n### TABLE 2: 期間一貫性（rsi_bb_reversal_short × atr_tp30_sl20）\n")

    # レジーム別に集計
    for regime in ['uptrend', 'downtrend', 'range']:
        print(f"\n#### {regime.capitalize()}\n")

        # 銘柄別にPASS期間を集計
        symbol_passes = defaultdict(list)

        for (symbol, period), test_results in data.items():
            result = get_oos_result(test_results, regime, template1, exit1)
            if result and result.get('metrics', {}).get('total_pnl', 0) > 0:
                symbol_passes[symbol].append(period)

        # 2期間以上PASSした銘柄のみ表示
        consistent_symbols = {s: periods for s, periods in symbol_passes.items() if len(periods) >= 2}

        if consistent_symbols:
            print("| 銘柄 | PASS期間数 | 期間 |")
            print("|------|-----------|------|")
            for symbol in sorted(consistent_symbols.keys()):
                periods = sorted(consistent_symbols[symbol])
                print(f"| {symbol} | {len(periods)} | {', '.join(periods)} |")
        else:
            print("**2期間以上PASSした銘柄なし**")

    # TABLE 3: bb_volume_reversal_short × atr_tp30_sl20
    print("\n### TABLE 3: bb_volume_reversal_short × atr_tp30_sl20 の銘柄×期間×レジーム詳細\n")
    print("| 銘柄 | 期間 | Uptrend | Downtrend | Range |")
    print("|------|------|---------|-----------|-------|")

    template2 = "bb_volume_reversal_short"
    exit2 = "atr_tp30_sl20"

    for (symbol, period), test_results in sorted(data.items()):
        uptrend = get_oos_result(test_results, 'uptrend', template2, exit2)
        downtrend = get_oos_result(test_results, 'downtrend', template2, exit2)
        range_regime = get_oos_result(test_results, 'range', template2, exit2)

        print(f"| {symbol} | {period} | {format_cell(uptrend)} | {format_cell(downtrend)} | {format_cell(range_regime)} |")

    # TABLE 4: 全体サマリー
    print("\n### TABLE 4: 全体サマリー\n")

    # 両テンプレートを集計
    templates = [
        (template1, exit1, "rsi_bb_reversal_short × atr_tp30_sl20"),
        (template2, exit2, "bb_volume_reversal_short × atr_tp30_sl20")
    ]

    for template, exit_profile, label in templates:
        print(f"\n#### {label}\n")

        # 銘柄別PASS数
        symbol_pass_count = defaultdict(int)
        # レジーム別PASS数
        regime_pass_count = defaultdict(int)
        # 期間別PASS数
        period_pass_count = defaultdict(int)

        for (symbol, period), test_results in data.items():
            for regime in ['uptrend', 'downtrend', 'range']:
                result = get_oos_result(test_results, regime, template, exit_profile)
                if result and result.get('metrics', {}).get('total_pnl', 0) > 0:
                    symbol_pass_count[symbol] += 1
                    regime_pass_count[regime] += 1
                    period_pass_count[period] += 1

        # 銘柄別PASS数（降順）
        print("**銘柄別PASS数（全レジーム・全期間合計）**\n")
        print("| 銘柄 | PASS数 |")
        print("|------|--------|")
        for symbol, count in sorted(symbol_pass_count.items(), key=lambda x: (-x[1], x[0])):
            print(f"| {symbol} | {count} |")

        # レジーム別PASS数
        print("\n**レジーム別PASS数**\n")
        print("| レジーム | PASS数 |")
        print("|---------|--------|")
        for regime in ['uptrend', 'downtrend', 'range']:
            print(f"| {regime.capitalize()} | {regime_pass_count[regime]} |")

        # 期間別PASS数
        print("\n**期間別PASS数**\n")
        print("| 期間 | PASS数 |")
        print("|------|--------|")
        for period in sorted(period_pass_count.keys()):
            print(f"| {period} | {period_pass_count[period]} |")

if __name__ == '__main__':
    main()
