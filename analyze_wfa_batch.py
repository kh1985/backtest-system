"""
WFAバッチ結果の集約分析スクリプト

Modal WFAバッチ結果（results/batch/[RUN_ID]/wfa/*.json）を読み込み、
ROBUST率、平均PnL、exit profile別の統計を出力する。
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_wfa_batch(batch_dir: str, regime: str = "uptrend", cr_threshold: float = 0.6):
    """WFAバッチ結果を分析"""
    batch_path = Path(batch_dir)
    wfa_dir = batch_path / "wfa"

    if not wfa_dir.exists():
        print(f"❌ WFAディレクトリが見つかりません: {wfa_dir}")
        return

    wfa_files = sorted(wfa_dir.glob("*_wfa.json"))
    if not wfa_files:
        print(f"❌ WFAファイルが見つかりません: {wfa_dir}")
        return

    print(f"📊 WFA結果分析: {len(wfa_files)}銘柄")
    print(f"   対象レジーム: {regime}")
    print(f"   CR閾値: {cr_threshold}")
    print()

    # 集約用
    results = []
    exit_stats = defaultdict(lambda: {"count": 0, "robust": 0, "pnl_sum": 0.0})

    for wfa_file in wfa_files:
        with open(wfa_file) as f:
            data = json.load(f)

        symbol = data["symbol"]
        aggregate = data.get("aggregate", {})

        # 指定レジームのCR取得
        cr = aggregate.get("consistency_ratio", {}).get(regime, 0.0)
        wfe = aggregate.get("wfe", {}).get(regime, 0.0)
        oos_pnl = aggregate.get("stitched_oos_pnl", {}).get(regime, 0.0)
        stability = aggregate.get("strategy_stability", {}).get(regime, 0.0)

        # Exit profileの集計（全フォールド）
        exit_counts = defaultdict(int)
        for fold in data.get("folds", []):
            oos_regime = fold.get("oos_results", {}).get(regime, {})
            exit_profile = oos_regime.get("exit_profile", "unknown")
            exit_counts[exit_profile] += 1

        # 最頻Exit profile
        if exit_counts:
            dominant_exit = max(exit_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_exit = "unknown"

        # ROBUST判定
        is_robust = cr >= cr_threshold

        results.append({
            "symbol": symbol,
            "cr": cr,
            "wfe": wfe,
            "oos_pnl": oos_pnl,
            "stability": stability,
            "robust": is_robust,
            "dominant_exit": dominant_exit
        })

        # Exit統計更新
        exit_stats[dominant_exit]["count"] += 1
        if is_robust:
            exit_stats[dominant_exit]["robust"] += 1
        exit_stats[dominant_exit]["pnl_sum"] += oos_pnl

    # 結果表示
    print(f"{'銘柄':<12} {'CR':>6} {'WFE':>7} {'OOS PnL':>9} {'Stab':>6} {'Exit':>15} {'判定':>6}")
    print("-" * 75)

    robust_count = 0
    total_pnl = 0.0

    for r in sorted(results, key=lambda x: x["cr"], reverse=True):
        status = "PASS" if r["robust"] else "FAIL"
        if r["robust"]:
            robust_count += 1
        total_pnl += r["oos_pnl"]

        print(f"{r['symbol']:<12} {r['cr']*100:>5.0f}% {r['wfe']:>7.2f} "
              f"{r['oos_pnl']:>8.1f}% {r['stability']*100:>5.0f}% "
              f"{r['dominant_exit']:>15} {status:>6}")

    print("-" * 75)
    print(f"ROBUST: {robust_count}/{len(results)} ({robust_count/len(results)*100:.0f}%)")
    print(f"平均PnL: {total_pnl/len(results):.1f}%")
    print()

    # Exit profile別統計
    print("📈 Exit Profile別統計:")
    print(f"{'Exit Profile':<20} {'使用回数':>8} {'ROBUST':>8} {'平均PnL':>10}")
    print("-" * 50)

    for exit_name, stats in sorted(exit_stats.items(), key=lambda x: x[1]["robust"], reverse=True):
        avg_pnl = stats["pnl_sum"] / stats["count"] if stats["count"] > 0 else 0.0
        robust_pct = stats["robust"] / stats["count"] * 100 if stats["count"] > 0 else 0.0
        print(f"{exit_name:<20} {stats['count']:>8} "
              f"{stats['robust']:>3}/{stats['count']:<3} ({robust_pct:>3.0f}%) "
              f"{avg_pnl:>8.1f}%")

    print()

    # ROBUST銘柄リスト
    print("✅ ROBUST銘柄:")
    robust_symbols = [r["symbol"] for r in results if r["robust"]]
    if robust_symbols:
        print(", ".join(robust_symbols))
    else:
        print("なし")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python3 analyze_wfa_batch.py <batch_dir> [regime] [cr_threshold]")
        print("例: python3 analyze_wfa_batch.py results/batch/20260211_153124_wfa uptrend 0.6")
        sys.exit(1)

    batch_dir = sys.argv[1]
    regime = sys.argv[2] if len(sys.argv) > 2 else "uptrend"
    cr_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6

    analyze_wfa_batch(batch_dir, regime, cr_threshold)
