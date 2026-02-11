#!/usr/bin/env python3
"""
OOS結果分析スクリプト v2
test_resultsに記録されている実際のテンプレート×exit_profileの組み合わせを表示
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
    parts = filename.replace('.json', '').split('_')
    symbol = parts[0]
    period = parts[1]
    return symbol, period

def format_cell(result: dict) -> str:
    """セルのフォーマット: PASS +X.X% (Ntrades) or FAIL -X.X% (Ntrades)"""
    metrics = result.get('metrics', {})
    pnl = metrics.get('total_pnl', 0)
    trades = metrics.get('trades', 0)

    if pnl > 0:
        return f"PASS +{pnl:.1f}% ({trades})"
    else:
        return f"FAIL {pnl:.1f}% ({trades})"

def get_result_info(result: dict) -> str:
    """テンプレート名とexit_profileを含む情報を返す"""
    template = result.get('template', 'N/A')
    exit_profile = result.get('exit_profile', 'N/A')
    metrics = result.get('metrics', {})
    pnl = metrics.get('total_pnl', 0)
    trades = metrics.get('trades', 0)

    status = "PASS" if pnl > 0 else "FAIL"
    return f"{template} × {exit_profile}: {status} {pnl:+.1f}% ({trades})"

def main():
    # データ収集
    data = {}  # {(symbol, period): test_results}

    # 全ファイル読み込み
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith('.json'):
            continue

        filepath = RESULTS_DIR / filename
        symbol, period = parse_filename(filename)

        with open(filepath, 'r') as f:
            content = json.load(f)

        test_results = content.get('test_results', {})
        data[(symbol, period)] = test_results

    # === 実際にテストされた組み合わせを確認 ===
    print("### 実際にテストされたテンプレート×Exit Profile の組み合わせ（サンプル10件）\n")

    count = 0
    for (symbol, period), test_results in sorted(data.items()):
        if count >= 10:
            break

        print(f"\n**{symbol} {period}**\n")
        for regime in ['uptrend', 'downtrend', 'range']:
            if regime in test_results:
                print(f"- {regime}: {get_result_info(test_results[regime])}")

        count += 1

    # === テンプレート×Exit組み合わせの統計 ===
    print("\n\n### テンプレート×Exit Profile の使用統計\n")

    combo_count = defaultdict(int)
    regime_combo_count = defaultdict(lambda: defaultdict(int))

    for (symbol, period), test_results in data.items():
        for regime, result in test_results.items():
            template = result.get('template', 'N/A')
            exit_profile = result.get('exit_profile', 'N/A')
            combo = f"{template} × {exit_profile}"
            combo_count[combo] += 1
            regime_combo_count[regime][combo] += 1

    print("**全レジーム・全銘柄・全期間での使用回数**\n")
    print("| テンプレート × Exit Profile | 使用回数 |")
    print("|----------------------------|----------|")
    for combo, count in sorted(combo_count.items(), key=lambda x: -x[1]):
        print(f"| {combo} | {count} |")

    for regime in ['uptrend', 'downtrend', 'range']:
        print(f"\n**{regime.capitalize()}での使用回数**\n")
        print("| テンプレート × Exit Profile | 使用回数 |")
        print("|----------------------------|----------|")
        for combo, count in sorted(regime_combo_count[regime].items(), key=lambda x: -x[1]):
            print(f"| {combo} | {count} |")

    # === 特定の組み合わせでのPASS/FAIL集計 ===
    print("\n\n### 特定テンプレート×Exit Profileの詳細分析\n")

    # 頻出上位3組み合わせを分析
    top_combos = sorted(combo_count.items(), key=lambda x: -x[1])[:5]

    for combo_str, _ in top_combos:
        # コンボ文字列を分解
        parts = combo_str.split(' × ')
        if len(parts) != 2:
            continue
        template, exit_profile = parts[0], parts[1]

        print(f"\n#### {combo_str}\n")
        print("| 銘柄 | 期間 | Uptrend | Downtrend | Range |")
        print("|------|------|---------|-----------|-------|")

        for (symbol, period), test_results in sorted(data.items()):
            cells = []
            for regime in ['uptrend', 'downtrend', 'range']:
                if regime in test_results:
                    result = test_results[regime]
                    if result.get('template') == template and result.get('exit_profile') == exit_profile:
                        cells.append(format_cell(result))
                    else:
                        cells.append("N/A")
                else:
                    cells.append("N/A")

            print(f"| {symbol} | {period} | {cells[0]} | {cells[1]} | {cells[2]} |")

        # この組み合わせでの統計
        pass_count = defaultdict(int)
        fail_count = defaultdict(int)

        for (symbol, period), test_results in data.items():
            for regime, result in test_results.items():
                if result.get('template') == template and result.get('exit_profile') == exit_profile:
                    pnl = result.get('metrics', {}).get('total_pnl', 0)
                    if pnl > 0:
                        pass_count[regime] += 1
                    else:
                        fail_count[regime] += 1

        print(f"\n**レジーム別 PASS/FAIL 統計**\n")
        print("| レジーム | PASS | FAIL | 合計 | PASS率 |")
        print("|---------|------|------|------|--------|")
        for regime in ['uptrend', 'downtrend', 'range']:
            total = pass_count[regime] + fail_count[regime]
            pass_rate = (pass_count[regime] / total * 100) if total > 0 else 0
            print(f"| {regime.capitalize()} | {pass_count[regime]} | {fail_count[regime]} | {total} | {pass_rate:.1f}% |")

if __name__ == '__main__':
    main()
