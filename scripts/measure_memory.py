"""
メモリ使用量測定スクリプト

共有メモリ修正の効果を検証する。
1銘柄 × 指定ワーカー数でグリッドサーチを実行し、
メイン+子プロセスのピークRSSを計測。

使い方:
  python scripts/measure_memory.py --workers 8
"""

import sys
import os
import resource
import time
import copy
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.batch_optimize import (
    load_symbol_data, prepare_exec_df, generate_all_configs,
    INPUTDATA_DIR, TARGET_REGIMES,
)
from optimizer.grid import GridSearchOptimizer


def get_peak_rss_mb() -> float:
    """現在プロセスのピークRSS (MB)"""
    # macOS: ru_maxrss はバイト単位
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def get_children_peak_rss_mb() -> float:
    """子プロセスのピークRSS (MB)"""
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="メモリ使用量測定")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--period", type=str, default="20250201-20260130")
    parser.add_argument("--exec-tf", type=str, default="1m")
    parser.add_argument("--htf", type=str, default="15m")
    parser.add_argument("--exit-profiles", type=str, default="fixed")
    args = parser.parse_args()

    print(f"=== メモリ測定: {args.symbol} {args.exec_tf}/{args.htf} workers={args.workers} ===")
    print(f"開始前 RSS: {get_peak_rss_mb():.1f} MB")

    # データ読み込み
    tf_dict = load_symbol_data(
        args.symbol, args.period, args.exec_tf, args.htf, INPUTDATA_DIR
    )
    exec_df = prepare_exec_df(tf_dict, args.exec_tf, args.htf)
    print(f"データ読込後 RSS: {get_peak_rss_mb():.1f} MB  ({len(exec_df)} bars)")

    # exit_profiles
    from optimizer.exit_profiles import get_profiles
    profiles = get_profiles(args.exit_profiles)

    all_configs = generate_all_configs(exit_profiles=profiles)
    print(f"Config数: {len(all_configs)} ({len(all_configs) * len(TARGET_REGIMES)} 組み合わせ)")

    # グリッドサーチ実行
    optimizer = GridSearchOptimizer(top_n_results=10)
    configs = copy.deepcopy(all_configs)

    t0 = time.time()
    result_set = optimizer.run(
        df=exec_df,
        configs=configs,
        target_regimes=TARGET_REGIMES,
        n_workers=args.workers,
    )
    elapsed = time.time() - t0

    # 結果
    self_peak = get_peak_rss_mb()
    children_peak = get_children_peak_rss_mb()

    print(f"\n=== 結果 ===")
    print(f"実行時間: {elapsed:.1f}s")
    print(f"メインプロセス ピークRSS: {self_peak:.1f} MB")
    print(f"子プロセス ピークRSS: {children_peak:.1f} MB")
    print(f"合計推定ピーク: {self_peak + children_peak * args.workers:.1f} MB")
    print(f"結果数: {result_set.total_combinations} 組み合わせ")
    if result_set.best:
        print(f"ベスト: {result_set.best.template_name} "
              f"score={result_set.best.composite_score:.4f}")


if __name__ == "__main__":
    main()
