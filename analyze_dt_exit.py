#!/usr/bin/env python3
"""
DT戦略のexit profile別ROBUST率集計
"""
import json
from pathlib import Path
from collections import defaultdict

# WFA結果ディレクトリ
wfa_dir = Path("results/batch/20260211_153140_wfa/wfa")

# Exit profile別の集計
exit_stats = defaultdict(lambda: {"robust": 0, "total": 0, "pnl_sum": 0.0, "symbols": []})

for wfa_file in sorted(wfa_dir.glob("*_wfa.json")):
    symbol = wfa_file.stem.replace("_wfa", "")

    with open(wfa_file) as f:
        data = json.load(f)

    # フォールド別のexit profile選択状況を集計
    fold_exits = []
    for fold in data["folds"]:
        if "downtrend" in fold["selected"]:
            exit_profile = fold["selected"]["downtrend"]["exit_profile"]
            fold_exits.append(exit_profile)

    # 集計サマリー
    agg = data["aggregate"]
    cr = agg["consistency_ratio"]["downtrend"]
    oos_pnl = agg["stitched_oos_pnl"]["downtrend"]
    is_robust = cr >= 0.6

    # フォールド別のexit使用頻度
    from collections import Counter
    exit_counts = Counter(fold_exits)

    # 最頻出exit
    if exit_counts:
        most_common_exit = exit_counts.most_common(1)[0][0]

        # Exit別統計に追加
        exit_stats[most_common_exit]["total"] += 1
        if is_robust:
            exit_stats[most_common_exit]["robust"] += 1
            exit_stats[most_common_exit]["symbols"].append(symbol)
        exit_stats[most_common_exit]["pnl_sum"] += oos_pnl

    print(f"{symbol:12s} CR={cr:.0%} OOS_PnL={oos_pnl:+.2f}% {'✓' if is_robust else '✗'} exits={dict(exit_counts)}")

print("\n" + "="*80)
print("Exit Profile別 ROBUST率集計")
print("="*80)

for exit_name in sorted(exit_stats.keys()):
    stats = exit_stats[exit_name]
    robust_rate = stats["robust"] / stats["total"] if stats["total"] > 0 else 0
    avg_pnl = stats["pnl_sum"] / stats["total"] if stats["total"] > 0 else 0

    print(f"\n{exit_name}")
    print(f"  ROBUST率: {robust_rate:.0%} ({stats['robust']}/{stats['total']})")
    print(f"  平均PnL:  {avg_pnl:+.2f}%")
    print(f"  ROBUST銘柄: {', '.join(stats['symbols']) if stats['symbols'] else 'なし'}")

print("\n" + "="*80)
print("全体サマリー")
print("="*80)
total_symbols = sum(s["total"] for s in exit_stats.values())
total_robust = sum(s["robust"] for s in exit_stats.values())
overall_pnl = sum(s["pnl_sum"] for s in exit_stats.values()) / total_symbols if total_symbols > 0 else 0

print(f"全体ROBUST率: {total_robust/total_symbols:.0%} ({total_robust}/{total_symbols})")
print(f"全体平均PnL:  {overall_pnl:+.2f}%")
