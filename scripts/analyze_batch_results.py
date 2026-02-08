#!/usr/bin/env python3
"""
バッチ最適化結果の解析スクリプト
85個のJSONファイルを解析し、テンプレート×Exit×レジーム別のPASS率テーブルを出力
"""
import json
import os
import glob
from collections import defaultdict

BASE_DIR = "/Users/kenjihachiya/Desktop/work/development/backtest-system/results/batch/20260206_221008/optimization/"

# PASS基準
MIN_TEST_TRADES = 20
MIN_TEST_PNL = 0.0  # test_pnl_pct > 0

# 実用レベル基準
MIN_SAMPLE_SIZE = 10
MIN_PASS_RATE = 50.0


def load_all_results():
    """全JSONファイルを読み込み、フラットなレコードリストを返す"""
    records = []
    files = sorted(glob.glob(os.path.join(BASE_DIR, "*.json")))

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        symbol = data["symbol"]
        period = data["period"]
        test_results = data.get("test_results", {})

        if not isinstance(test_results, dict):
            continue

        for regime, result in test_results.items():
            if not isinstance(result, dict):
                continue
            metrics = result.get("metrics", {})
            records.append({
                "symbol": symbol,
                "period": period,
                "regime": regime,
                "template": result.get("template", "unknown"),
                "exit_profile": result.get("exit_profile", "unknown"),
                "params": result.get("params", {}),
                "trades": metrics.get("trades", 0),
                "pnl": metrics.get("total_pnl", 0),
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "max_dd": metrics.get("max_dd", 0),
                "sharpe": metrics.get("sharpe", 0),
                "score": result.get("score", 0),
            })

    return records


def is_pass(record):
    """PASS判定: test_pnl > 0 かつ trades >= 20"""
    return record["pnl"] > MIN_TEST_PNL and record["trades"] >= MIN_TEST_TRADES


def print_separator(char="=", width=120):
    print(char * width)


def table1_template_exit_regime(records):
    """テーブル1: テンプレート×Exit×レジーム別 PASS率"""
    print()
    print_separator()
    print("## テーブル1: テンプレート×Exit×レジーム別 PASS率")
    print_separator()
    print()
    print("PASS基準: test_pnl > 0% かつ test_trades >= 20")
    print("★ = 実用レベル（PASS率 >= 50% かつ n >= 10）")
    print()

    # 集計
    stats = defaultdict(lambda: {"n": 0, "pass": 0})
    for r in records:
        key = (r["template"], r["exit_profile"], r["regime"])
        stats[key]["n"] += 1
        if is_pass(r):
            stats[key]["pass"] += 1

    # ソート: テンプレート名 → Exit → レジーム
    sorted_keys = sorted(stats.keys())

    # ヘッダー
    print(f"| {'テンプレート':<35} | {'Exit':<16} | {'レジーム':<12} | {'n':>3} | {'PASS':>4} | {'率%':>6} | {'判定':>4} |")
    print(f"|{'-'*37}|{'-'*18}|{'-'*14}|{'-'*5}|{'-'*6}|{'-'*8}|{'-'*6}|")

    practical_combos = []
    for key in sorted_keys:
        template, exit_p, regime = key
        s = stats[key]
        rate = (s["pass"] / s["n"] * 100) if s["n"] > 0 else 0
        is_practical = rate >= MIN_PASS_RATE and s["n"] >= MIN_SAMPLE_SIZE
        mark = "★" if is_practical else ""
        if is_practical:
            practical_combos.append((template, exit_p, regime, s["n"], s["pass"], rate))
        print(f"| {template:<35} | {exit_p:<16} | {regime:<12} | {s['n']:>3} | {s['pass']:>4} | {rate:>5.1f}% | {mark:>3} |")

    # 実用レベルまとめ
    if practical_combos:
        print()
        print("### 実用レベル（★）の組み合わせ:")
        print()
        print(f"| {'テンプレート':<35} | {'Exit':<16} | {'レジーム':<12} | {'n':>3} | {'PASS':>4} | {'率%':>6} |")
        print(f"|{'-'*37}|{'-'*18}|{'-'*14}|{'-'*5}|{'-'*6}|{'-'*8}|")
        for template, exit_p, regime, n, p, rate in sorted(practical_combos, key=lambda x: -x[5]):
            print(f"| {template:<35} | {exit_p:<16} | {regime:<12} | {n:>3} | {p:>4} | {rate:>5.1f}% |")
    else:
        print()
        print("### 実用レベル（★）の組み合わせ: なし")


def table1b_template_regime_aggregated(records):
    """テーブル1b: テンプレート×レジーム別 PASS率（Exit統合）"""
    print()
    print_separator()
    print("## テーブル1b: テンプレート×レジーム別 PASS率（Exit統合）")
    print_separator()
    print()

    stats = defaultdict(lambda: {"n": 0, "pass": 0})
    for r in records:
        key = (r["template"], r["regime"])
        stats[key]["n"] += 1
        if is_pass(r):
            stats[key]["pass"] += 1

    # テンプレート一覧
    templates = sorted(set(k[0] for k in stats.keys()))
    regimes = ["uptrend", "downtrend", "range"]

    print(f"| {'テンプレート':<35} |", end="")
    for regime in regimes:
        print(f" {regime:^18} |", end="")
    print(f" {'合計':^18} |")

    print(f"|{'-'*37}|", end="")
    for _ in regimes:
        print(f"{'-'*20}|", end="")
    print(f"{'-'*20}|")

    for template in templates:
        print(f"| {template:<35} |", end="")
        total_n = 0
        total_pass = 0
        for regime in regimes:
            s = stats.get((template, regime), {"n": 0, "pass": 0})
            rate = (s["pass"] / s["n"] * 100) if s["n"] > 0 else 0
            total_n += s["n"]
            total_pass += s["pass"]
            mark = "★" if rate >= MIN_PASS_RATE and s["n"] >= MIN_SAMPLE_SIZE else " "
            print(f" {s['pass']:>2}/{s['n']:>2} ({rate:>5.1f}%){mark} |", end="")
        total_rate = (total_pass / total_n * 100) if total_n > 0 else 0
        print(f" {total_pass:>2}/{total_n:>2} ({total_rate:>5.1f}%)  |")


def table2_regime_summary(records):
    """テーブル2: レジーム別サマリー"""
    print()
    print_separator()
    print("## テーブル2: レジーム別サマリー")
    print_separator()
    print()

    stats = defaultdict(lambda: {"n": 0, "pass": 0, "pnl_sum": 0, "pnl_list": []})
    for r in records:
        s = stats[r["regime"]]
        s["n"] += 1
        if is_pass(r):
            s["pass"] += 1
        s["pnl_sum"] += r["pnl"]
        s["pnl_list"].append(r["pnl"])

    regimes = ["uptrend", "downtrend", "range"]

    print(f"| {'レジーム':<12} | {'n':>4} | {'PASS':>4} | {'PASS率':>7} | {'平均PnL':>8} | {'PnL中央値':>10} |")
    print(f"|{'-'*14}|{'-'*6}|{'-'*6}|{'-'*9}|{'-'*10}|{'-'*12}|")

    for regime in regimes:
        s = stats.get(regime, {"n": 0, "pass": 0, "pnl_sum": 0, "pnl_list": []})
        rate = (s["pass"] / s["n"] * 100) if s["n"] > 0 else 0
        avg_pnl = (s["pnl_sum"] / s["n"]) if s["n"] > 0 else 0
        sorted_pnl = sorted(s["pnl_list"])
        median_pnl = sorted_pnl[len(sorted_pnl) // 2] if sorted_pnl else 0
        print(f"| {regime:<12} | {s['n']:>4} | {s['pass']:>4} | {rate:>5.1f}%  | {avg_pnl:>7.2f}% | {median_pnl:>9.2f}% |")

    # 全体
    all_n = sum(s["n"] for s in stats.values())
    all_pass = sum(s["pass"] for s in stats.values())
    all_rate = (all_pass / all_n * 100) if all_n > 0 else 0
    all_pnl = [r["pnl"] for r in records]
    avg_all = sum(all_pnl) / len(all_pnl) if all_pnl else 0
    sorted_all = sorted(all_pnl)
    median_all = sorted_all[len(sorted_all) // 2] if sorted_all else 0
    print(f"|{'-'*14}|{'-'*6}|{'-'*6}|{'-'*9}|{'-'*10}|{'-'*12}|")
    print(f"| {'合計':<12} | {all_n:>4} | {all_pass:>4} | {all_rate:>5.1f}%  | {avg_all:>7.2f}% | {median_all:>9.2f}% |")


def table3_template_summary(records):
    """テーブル3: テンプレート別サマリー"""
    print()
    print_separator()
    print("## テーブル3: テンプレート別サマリー")
    print_separator()
    print()

    stats = defaultdict(lambda: {"n": 0, "pass": 0, "pnl_sum": 0})
    for r in records:
        s = stats[r["template"]]
        s["n"] += 1
        if is_pass(r):
            s["pass"] += 1
        s["pnl_sum"] += r["pnl"]

    # PASS率で降順ソート
    sorted_templates = sorted(stats.items(), key=lambda x: -(x[1]["pass"] / x[1]["n"]) if x[1]["n"] > 0 else 0)

    print(f"| {'テンプレート':<35} | {'n':>4} | {'PASS':>4} | {'PASS率':>7} | {'平均PnL':>8} |")
    print(f"|{'-'*37}|{'-'*6}|{'-'*6}|{'-'*9}|{'-'*10}|")

    for template, s in sorted_templates:
        rate = (s["pass"] / s["n"] * 100) if s["n"] > 0 else 0
        avg_pnl = (s["pnl_sum"] / s["n"]) if s["n"] > 0 else 0
        print(f"| {template:<35} | {s['n']:>4} | {s['pass']:>4} | {rate:>5.1f}%  | {avg_pnl:>7.2f}% |")


def table4_symbol_matrix(records):
    """テーブル4: 銘柄×レジーム×テンプレート マトリクス"""
    print()
    print_separator()
    print("## テーブル4: 銘柄別×レジーム マトリクス（全期間ベスト）")
    print_separator()
    print()
    print("各銘柄×レジームで、全期間のテスト結果を表示")
    print("PASS = pnl > 0% かつ trades >= 20")
    print()

    # 銘柄×レジーム×期間ごとにまとめる
    regimes = ["uptrend", "downtrend", "range"]
    symbols = sorted(set(r["symbol"] for r in records))
    periods = sorted(set(r["period"] for r in records))

    for regime in regimes:
        print()
        print(f"### {regime}")
        print()

        # ヘッダー
        period_labels = []
        for p in periods:
            year = p[:4]
            period_labels.append(year)

        print(f"| {'銘柄':<12} |", end="")
        for label in period_labels:
            print(f" {label + ' テンプレート':<30} | {'Exit':<16} | {'PnL%':>7} | {'Tr':>3} | {'判定':>4} |", end="")
        print()

        sep = f"|{'-'*14}|"
        for _ in periods:
            sep += f"{'-'*32}|{'-'*18}|{'-'*9}|{'-'*5}|{'-'*6}|"
        print(sep)

        for symbol in symbols:
            print(f"| {symbol:<12} |", end="")
            for period in periods:
                # この銘柄×期間×レジームのレコードを検索
                matching = [r for r in records if r["symbol"] == symbol and r["period"] == period and r["regime"] == regime]
                if matching:
                    r = matching[0]
                    passed = is_pass(r)
                    mark = "PASS" if passed else "FAIL"
                    print(f" {r['template']:<30} | {r['exit_profile']:<16} | {r['pnl']:>6.1f}% | {r['trades']:>3} | {mark:<4} |", end="")
                else:
                    print(f" {'--- (データなし)':<30} | {'---':<16} | {'---':>7} | {'---':>3} | {'---':>4} |", end="")
            print()


def table5_cross_period_consistency(records):
    """テーブル5: 銘柄×レジーム 期間横断PASS一貫性"""
    print()
    print_separator()
    print("## テーブル5: 期間横断PASS一貫性（銘柄×レジーム）")
    print_separator()
    print()
    print("各銘柄×レジームで3期間中のPASS数を集計")
    print()

    regimes = ["uptrend", "downtrend", "range"]
    symbols = sorted(set(r["symbol"] for r in records))
    periods = sorted(set(r["period"] for r in records))

    print(f"| {'銘柄':<12} |", end="")
    for regime in regimes:
        print(f" {regime:^14} |", end="")
    print()

    sep = f"|{'-'*14}|"
    for _ in regimes:
        sep += f"{'-'*16}|"
    print(sep)

    regime_totals = defaultdict(lambda: {"n": 0, "pass": 0})

    for symbol in symbols:
        print(f"| {symbol:<12} |", end="")
        for regime in regimes:
            matching = [r for r in records if r["symbol"] == symbol and r["regime"] == regime]
            n_periods = len(matching)
            n_pass = sum(1 for r in matching if is_pass(r))
            regime_totals[regime]["n"] += n_periods
            regime_totals[regime]["pass"] += n_pass

            if n_periods == 0:
                print(f" {'---':^14} |", end="")
            else:
                consistency = f"{n_pass}/{n_periods}"
                if n_pass == n_periods and n_periods >= 2:
                    mark = " ★★"
                elif n_pass >= 2:
                    mark = " ★"
                else:
                    mark = ""
                print(f" {consistency + mark:^14} |", end="")
        print()

    # 合計行
    print(f"|{'-'*14}|", end="")
    for _ in regimes:
        print(f"{'-'*16}|", end="")
    print()

    print(f"| {'合計PASS率':<12} |", end="")
    for regime in regimes:
        s = regime_totals[regime]
        rate = (s["pass"] / s["n"] * 100) if s["n"] > 0 else 0
        print(f" {s['pass']}/{s['n']} ({rate:.1f}%) ", end="|")
    print()

    # 2期間以上PASS銘柄のリスト
    print()
    print("### 2期間以上PASSした銘柄:")
    for regime in regimes:
        consistent = []
        for symbol in symbols:
            matching = [r for r in records if r["symbol"] == symbol and r["regime"] == regime]
            n_pass = sum(1 for r in matching if is_pass(r))
            if n_pass >= 2:
                consistent.append(f"{symbol}({n_pass}/{len(matching)})")
        if consistent:
            print(f"  {regime}: {', '.join(consistent)}")
        else:
            print(f"  {regime}: なし")


def table6_exit_profile_summary(records):
    """テーブル6: Exit Profile別サマリー"""
    print()
    print_separator()
    print("## テーブル6: Exit Profile別サマリー")
    print_separator()
    print()

    stats = defaultdict(lambda: {"n": 0, "pass": 0, "pnl_sum": 0})
    for r in records:
        s = stats[r["exit_profile"]]
        s["n"] += 1
        if is_pass(r):
            s["pass"] += 1
        s["pnl_sum"] += r["pnl"]

    sorted_exits = sorted(stats.items(), key=lambda x: -(x[1]["pass"] / x[1]["n"]) if x[1]["n"] > 0 else 0)

    print(f"| {'Exit Profile':<20} | {'n':>4} | {'PASS':>4} | {'PASS率':>7} | {'平均PnL':>8} |")
    print(f"|{'-'*22}|{'-'*6}|{'-'*6}|{'-'*9}|{'-'*10}|")

    for exit_p, s in sorted_exits:
        rate = (s["pass"] / s["n"] * 100) if s["n"] > 0 else 0
        avg_pnl = (s["pnl_sum"] / s["n"]) if s["n"] > 0 else 0
        print(f"| {exit_p:<20} | {s['n']:>4} | {s['pass']:>4} | {rate:>5.1f}%  | {avg_pnl:>7.2f}% |")


def main():
    print("=" * 120)
    print("  バッチ最適化結果 解析レポート")
    print(f"  対象: {BASE_DIR}")
    print("=" * 120)

    records = load_all_results()
    print(f"\n読み込み件数: {len(records)} レコード（{len(set(r['symbol'] for r in records))}銘柄 × {len(set(r['period'] for r in records))}期間 × 3レジーム）")
    print(f"PASS基準: test_pnl > 0% かつ test_trades >= {MIN_TEST_TRADES}")

    # 全テーブル出力
    table2_regime_summary(records)
    table3_template_summary(records)
    table6_exit_profile_summary(records)
    table1b_template_regime_aggregated(records)
    table1_template_exit_regime(records)
    table5_cross_period_consistency(records)
    table4_symbol_matrix(records)

    print()
    print_separator()
    print("  レポート終了")
    print_separator()


if __name__ == "__main__":
    main()
