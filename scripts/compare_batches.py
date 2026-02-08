#!/usr/bin/env python3
"""
Step 6.5 vs Step 7b バッチ結果比較分析スクリプト
"""
import json
import glob
import os
from collections import defaultdict

# ===== 設定 =====
BATCH_STEP65 = "/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260206_200522/optimization/"
BATCH_STEP7B = "/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260206_221008/optimization/"

# PASS判定: test_pnl_pct > 0 AND test_trades >= 20
MIN_TRADES = 20

# 共通テンプレート
COMMON_TEMPLATES = ["rsi_bb_reversal_short", "bb_volume_reversal_short"]

# 全テンプレート（Step 7bの追加分含む）
ALL_TEMPLATES = [
    "rsi_bb_reversal_short", "rsi_bb_reversal_long",
    "bb_volume_reversal_short", "bb_volume_reversal_long",
    "rsi_bb_volume_reversal_short",
]

REGIMES = ["uptrend", "downtrend", "range"]

# 逆張り除外ルール
EXCLUDED_COMBOS = {
    # Uptrendでショート → 除外
    ("rsi_bb_reversal_short", "uptrend"),
    ("bb_volume_reversal_short", "uptrend"),
    ("rsi_bb_volume_reversal_short", "uptrend"),
    # Downtrendでロング → 除外
    ("rsi_bb_reversal_long", "downtrend"),
    ("bb_volume_reversal_long", "downtrend"),
}


def load_batch(batch_dir):
    """バッチ結果を読み込み。各ファイル(=1銘柄×1期間)から test_results を抽出"""
    records = []
    files = sorted(glob.glob(os.path.join(batch_dir, "*.json")))
    for fp in files:
        fname = os.path.basename(fp)
        if fname == "CLAUDE.md":
            continue
        with open(fp) as f:
            data = json.load(f)

        symbol = data.get("symbol", fname.split("_")[0])
        period = data.get("period", "")

        # test_results はレジーム別の dict
        test_results = data.get("test_results", {})
        if not isinstance(test_results, dict):
            continue

        for regime, entry in test_results.items():
            if not isinstance(entry, dict):
                continue
            template = entry.get("template", "")
            metrics = entry.get("metrics", {})
            trades = metrics.get("trades", 0)
            pnl = metrics.get("total_pnl", 0)
            exit_profile = entry.get("exit_profile", "")

            records.append({
                "symbol": symbol,
                "period": period,
                "regime": regime,
                "template": template,
                "exit_profile": exit_profile,
                "trades": trades,
                "pnl": pnl,
                "passed": pnl > 0 and trades >= MIN_TRADES,
            })
    return records


def calc_pass_rate(records, template, regime):
    """指定テンプレート×レジームのPASS率を計算"""
    filtered = [r for r in records if r["template"] == template and r["regime"] == regime]
    n = len(filtered)
    if n == 0:
        return 0, 0, 0.0
    passed = sum(1 for r in filtered if r["passed"])
    return n, passed, (passed / n * 100) if n > 0 else 0.0


def calc_pass_rate_by_exit(records, template, regime):
    """指定テンプレート×レジーム×Exit別のPASS率を計算"""
    filtered = [r for r in records if r["template"] == template and r["regime"] == regime]
    exit_groups = defaultdict(list)
    for r in filtered:
        exit_groups[r["exit_profile"]].append(r)

    results = {}
    for exit_p, recs in sorted(exit_groups.items()):
        n = len(recs)
        passed = sum(1 for r in recs if r["passed"])
        results[exit_p] = (n, passed, (passed / n * 100) if n > 0 else 0.0)
    return results


def print_section(title, level=1):
    prefix = "#" * level
    print(f"\n{prefix} {title}\n")


def main():
    # データ読み込み
    step65 = load_batch(BATCH_STEP65)
    step7b = load_batch(BATCH_STEP7B)

    print("=" * 80)
    print("# Step 6.5 vs Step 7b バッチ結果比較分析")
    print("=" * 80)

    # --- 基本情報 ---
    print_section("基本情報", 2)
    for label, records, batch_id in [
        ("Step 6.5", step65, "20260206_200522"),
        ("Step 7b", step7b, "20260206_221008"),
    ]:
        symbols = sorted(set(r["symbol"] for r in records))
        templates = sorted(set(r["template"] for r in records))
        total = len(records)
        passed = sum(1 for r in records if r["passed"])
        print(f"**{label}** (Run: {batch_id})")
        print(f"- テスト件数: {total} (銘柄×期間×レジーム)")
        print(f"- 銘柄数: {len(symbols)}")
        print(f"- テンプレート: {', '.join(templates)}")
        print(f"- 全体PASS: {passed}/{total} ({passed/total*100:.1f}%)")
        print()

    # ===== 1. 共通テンプレートのPASS率比較 =====
    print_section("1. 共通テンプレート PASS率比較", 2)
    print("PASS判定: `test_pnl > 0 AND test_trades >= 20`")
    print()
    print("Step 6.5: sigma=2.0/2.5/3.0, RSI 60-80, RVOL 2.0-4.0")
    print("Step 7b: sigma=2.0固定, RSI 60-75, RVOL 2.5-4.0 + Long追加 + rsi_bb_volume_reversal_short追加")
    print()

    # テーブルヘッダー
    print("| テンプレート | レジーム | Step 6.5 n | Step 6.5 PASS | Step 6.5 率 | Step 7b n | Step 7b PASS | Step 7b 率 | 差分 |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")

    for tmpl in COMMON_TEMPLATES:
        for regime in REGIMES:
            n65, p65, r65 = calc_pass_rate(step65, tmpl, regime)
            n7b, p7b, r7b = calc_pass_rate(step7b, tmpl, regime)
            diff = r7b - r65 if n65 > 0 and n7b > 0 else float("nan")
            diff_str = f"{diff:+.1f}pp" if not (diff != diff) else "N/A"
            print(f"| {tmpl} | {regime} | {n65} | {p65} | {r65:.1f}% | {n7b} | {p7b} | {r7b:.1f}% | {diff_str} |")

    # ===== 2. 全テンプレートに運用候補マーク =====
    print_section("2. 実運用フィルタリング（逆張り除外）", 2)
    print("除外ルール:")
    print("- Uptrendでショート系 → 除外（トレンドフォロー違反）")
    print("- Downtrendでロング系 → 除外（トレンドフォロー違反）")
    print()

    print("| テンプレート | レジーム | 判定 | 理由 |")
    print("|:---|:---|:---|:---|")
    for tmpl in ALL_TEMPLATES:
        for regime in REGIMES:
            if (tmpl, regime) in EXCLUDED_COMBOS:
                reason = "ショートinUptrend" if regime == "uptrend" else "ロングinDowntrend"
                print(f"| {tmpl} | {regime} | **除外** | {reason} |")
            else:
                print(f"| {tmpl} | {regime} | 候補 | - |")

    # ===== 3. 実運用候補PASS率テーブル（除外後） =====
    print_section("3. 実運用候補PASS率テーブル（Step 7b、除外後）", 2)
    print("PASS判定: `test_pnl > 0 AND test_trades >= 20`")
    print("目標: PASS率 >= 50% かつ n >= 10")
    print()

    # まずはテンプレート×レジーム集計
    print("### 3a. テンプレート×レジーム集計")
    print()
    print("| テンプレート | レジーム | n | PASS | 率 | 判定 |")
    print("|:---|:---|---:|---:|---:|:---|")

    candidates = []
    for tmpl in ALL_TEMPLATES:
        for regime in REGIMES:
            if (tmpl, regime) in EXCLUDED_COMBOS:
                continue
            n, passed, rate = calc_pass_rate(step7b, tmpl, regime)
            if n == 0:
                continue

            if rate >= 50 and n >= 10:
                status = "**GOAL**"
            elif rate >= 50 and n >= 5:
                status = "参考値(n<10)"
            elif n >= 5:
                status = "-"
            else:
                status = "n不足"

            print(f"| {tmpl} | {regime} | {n} | {passed} | {rate:.1f}% | {status} |")
            candidates.append((tmpl, regime, n, passed, rate, status))

    # テンプレート×Exit×レジーム の詳細
    print()
    print("### 3b. テンプレート×Exit×レジーム 詳細（候補のみ）")
    print()
    print("| テンプレート | レジーム | Exit | n | PASS | 率 |")
    print("|:---|:---|:---|---:|---:|---:|")

    for tmpl in ALL_TEMPLATES:
        for regime in REGIMES:
            if (tmpl, regime) in EXCLUDED_COMBOS:
                continue
            exit_rates = calc_pass_rate_by_exit(step7b, tmpl, regime)
            for exit_p, (n, passed, rate) in exit_rates.items():
                if n == 0:
                    continue
                print(f"| {tmpl} | {regime} | {exit_p} | {n} | {passed} | {rate:.1f}% |")

    # ===== 4. 銘柄別PASS一覧（実運用候補のみ） =====
    print_section("4. 銘柄別PASS一覧（Step 7b、実運用候補のみ）", 2)
    print("PASSした銘柄×期間のみ表示")
    print()

    for tmpl in ALL_TEMPLATES:
        for regime in REGIMES:
            if (tmpl, regime) in EXCLUDED_COMBOS:
                continue
            passed_records = [
                r for r in step7b
                if r["template"] == tmpl and r["regime"] == regime and r["passed"]
            ]
            if not passed_records:
                continue
            print(f"**{tmpl} / {regime}** ({len(passed_records)} PASS):")
            for r in sorted(passed_records, key=lambda x: -x["pnl"]):
                print(f"  - {r['symbol']} ({r['period']}): PnL={r['pnl']:+.2f}%, trades={r['trades']}, exit={r['exit_profile']}")
            print()

    # ===== 5. Step 6.5 との差分サマリー =====
    print_section("5. Step 6.5 → Step 7b 変更サマリー", 2)

    # 共通テンプレートの変化
    print("### 共通テンプレート（short系）の変化")
    print()
    for tmpl in COMMON_TEMPLATES:
        total_n65 = sum(calc_pass_rate(step65, tmpl, r)[0] for r in REGIMES)
        total_p65 = sum(calc_pass_rate(step65, tmpl, r)[1] for r in REGIMES)
        total_n7b = sum(calc_pass_rate(step7b, tmpl, r)[0] for r in REGIMES)
        total_p7b = sum(calc_pass_rate(step7b, tmpl, r)[1] for r in REGIMES)
        r65 = (total_p65 / total_n65 * 100) if total_n65 > 0 else 0
        r7b = (total_p7b / total_n7b * 100) if total_n7b > 0 else 0
        print(f"- **{tmpl}**: {total_p65}/{total_n65} ({r65:.1f}%) → {total_p7b}/{total_n7b} ({r7b:.1f}%)")
    print()

    # Step 7b 新規テンプレート
    new_templates = [t for t in ALL_TEMPLATES if t not in COMMON_TEMPLATES]
    if new_templates:
        print("### Step 7b 新規テンプレート")
        print()
        for tmpl in new_templates:
            for regime in REGIMES:
                if (tmpl, regime) in EXCLUDED_COMBOS:
                    continue
                n, passed, rate = calc_pass_rate(step7b, tmpl, regime)
                if n > 0:
                    print(f"- **{tmpl}** / {regime}: {passed}/{n} ({rate:.1f}%)")
        print()

    # ===== 6. 3期間一貫性チェック =====
    print_section("6. 3期間一貫性チェック（Step 7b）", 2)
    print("同一銘柄で3期間(2023, 2024, 2025)すべてPASSしたケース")
    print()

    # 銘柄×テンプレート×レジーム でグループ化
    from collections import defaultdict
    consistency = defaultdict(list)
    for r in step7b:
        if (r["template"], r["regime"]) in EXCLUDED_COMBOS:
            continue
        if r["passed"]:
            key = (r["symbol"], r["template"], r["regime"])
            consistency[key].append(r["period"])

    found_3period = False
    for (sym, tmpl, regime), periods in sorted(consistency.items()):
        if len(periods) >= 3:
            found_3period = True
            print(f"- **{sym}** / {tmpl} / {regime}: {len(periods)}期間PASS ({', '.join(sorted(periods))})")

    if not found_3period:
        print("3期間一貫PASSは見つかりませんでした。")
    print()

    # 2期間PASS
    print("2期間PASSのケース:")
    found_2period = False
    for (sym, tmpl, regime), periods in sorted(consistency.items()):
        if len(periods) == 2:
            found_2period = True
            print(f"- {sym} / {tmpl} / {regime}: {len(periods)}期間PASS ({', '.join(sorted(periods))})")

    if not found_2period:
        print("2期間PASSも見つかりませんでした。")


if __name__ == "__main__":
    main()
