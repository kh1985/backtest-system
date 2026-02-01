"""
既存の結果JSONにdata_periodを一括付与するマイグレーションスクリプト。

使い方:
    # ドライラン（変更なし、対象ファイルの確認のみ）
    python scripts/backfill_data_period.py --dry-run

    # 2025期間データ (20250201-20260130) を全originalファイルに付与
    python scripts/backfill_data_period.py --start 2025-02-01 --end 2026-01-30

    # 2024期間データ (20240201-20250131) を特定シンボルに付与
    python scripts/backfill_data_period.py --start 2024-02-01 --end 2025-01-31 --symbol BTCUSDT

    # 特定のタイムスタンプ範囲のファイルのみ対象
    python scripts/backfill_data_period.py --start 2025-02-01 --end 2026-01-30 --after 20260201_060000
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="既存の結果JSONにdata_periodを一括付与"
    )
    parser.add_argument("--start", help="データ期間の開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", help="データ期間の終了日 (YYYY-MM-DD)")
    parser.add_argument("--symbol", help="対象シンボル（省略時は全シンボル）")
    parser.add_argument(
        "--after",
        help="このタイムスタンプ以降のファイルのみ対象 (YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--before",
        help="このタイムスタンプ以前のファイルのみ対象 (YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="変更せずに対象ファイルを表示",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のdata_periodを上書き",
    )
    args = parser.parse_args()

    results_dir = Path("results")
    if not results_dir.exists():
        print("results/ ディレクトリが見つかりません")
        sys.exit(1)

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print("JSONファイルが見つかりません")
        sys.exit(1)

    targets = []
    skipped_has_dp = 0
    skipped_trimmed = 0
    skipped_filter = 0

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data_period が既にある場合はスキップ（--force 除く）
        if "data_period" in data and not args.force:
            skipped_has_dp += 1
            continue

        # trimmed はスキップ（既にdata_periodがあるはず）
        if data.get("data_source") == "trimmed" and not args.force:
            skipped_trimmed += 1
            continue

        # シンボルフィルタ
        if args.symbol and data.get("symbol") != args.symbol:
            skipped_filter += 1
            continue

        # タイムスタンプフィルタ
        ts = data.get("timestamp", "")
        if args.after and ts < args.after:
            skipped_filter += 1
            continue
        if args.before and ts > args.before:
            skipped_filter += 1
            continue

        targets.append((fp, data))

    print(f"対象: {len(targets)} ファイル")
    print(f"スキップ: data_period既存={skipped_has_dp}, trimmed={skipped_trimmed}, フィルタ={skipped_filter}")
    print()

    if not targets:
        print("対象ファイルがありません")
        return

    # 対象ファイル一覧
    for fp, data in targets[:20]:
        sym = data.get("symbol", "?")
        ts = data.get("timestamp", "?")
        print(f"  {sym} | {ts} | {fp.name}")
    if len(targets) > 20:
        print(f"  ... 他 {len(targets) - 20} ファイル")

    if args.dry_run:
        print("\n[ドライラン] 変更なし")
        return

    if not args.start or not args.end:
        print("\n--start と --end を指定してください")
        print("例: --start 2025-02-01 --end 2026-01-30")
        sys.exit(1)

    # 付与実行
    print(f"\ndata_period: {args.start} ~ {args.end} を付与中...")
    updated = 0
    for fp, data in targets:
        data["data_period"] = {
            "start": args.start,
            "end": args.end,
        }

        # data_period を results の前に配置
        ordered = {}
        for key in data:
            if key == "results":
                continue
            ordered[key] = data[key]
        ordered["results"] = data.get("results", [])

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
        updated += 1

    print(f"完了: {updated} ファイル更新")


if __name__ == "__main__":
    main()
