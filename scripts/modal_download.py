"""
Modal Volumeから最適化結果をダウンロードするスクリプト

使い方:
    # 最新のrun結果をダウンロード
    modal run scripts/modal_download.py

    # 特定のrun IDを指定
    modal run scripts/modal_download.py --run-id 20260206_143022

    # ダウンロード先を指定
    modal run scripts/modal_download.py --output-dir ./results/modal
"""

import json
from pathlib import Path

import modal

vol_results = modal.Volume.from_name("prism-results", create_if_missing=True)
app = modal.App("prism-download")


@app.function(
    cpu=1,
    memory=512,
    timeout=120,
    volumes={"/results": vol_results},
)
def list_runs() -> list:
    """Volume内のrun一覧を取得"""
    from pathlib import Path
    results_dir = Path("/results")
    runs = []
    if results_dir.exists():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir():
                opt_dir = d / "optimization"
                n_files = len(list(opt_dir.glob("*.json"))) if opt_dir.exists() else 0
                has_report = (d / "report.md").exists()
                runs.append({
                    "run_id": d.name,
                    "n_results": n_files,
                    "has_report": has_report,
                })
    return runs


@app.function(
    cpu=1,
    memory=1024,
    timeout=300,
    volumes={"/results": vol_results},
)
def download_run(run_id: str) -> dict:
    """指定runの全ファイルを読み込んで返す"""
    from pathlib import Path

    run_dir = Path(f"/results/{run_id}")
    if not run_dir.exists():
        return {"error": f"Run {run_id} が見つかりません"}

    files = {}

    # optimization/ 内のJSON
    opt_dir = run_dir / "optimization"
    if opt_dir.exists():
        for f in opt_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fh:
                files[f"optimization/{f.name}"] = fh.read()

    # ranking.json
    ranking_path = run_dir / "ranking.json"
    if ranking_path.exists():
        with open(ranking_path, "r", encoding="utf-8") as fh:
            files["ranking.json"] = fh.read()

    # report.md
    report_path = run_dir / "report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as fh:
            files["report.md"] = fh.read()

    # config.json
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            files["config.json"] = fh.read()

    return files


@app.local_entrypoint()
def main(
    run_id: str = "",
    output_dir: str = "",
):
    """結果をModal Volumeからローカルにダウンロード"""

    # run一覧を取得
    runs = list_runs.remote()

    if not runs:
        print("Modal Volume に結果がありません")
        return

    # run_id未指定 → 最新を使う
    if not run_id:
        latest = runs[-1]
        run_id = latest["run_id"]
        print(f"最新のrun: {run_id} ({latest['n_results']} ファイル)")
    else:
        found = [r for r in runs if r["run_id"] == run_id]
        if not found:
            print(f"Run ID '{run_id}' が見つかりません")
            print(f"利用可能なrun:")
            for r in runs:
                print(f"  {r['run_id']}: {r['n_results']} ファイル, レポート: {r['has_report']}")
            return

    # ダウンロード先
    if not output_dir:
        output_dir = f"results/batch/{run_id}"
    output_path = Path(output_dir)

    print(f"ダウンロード中: {run_id} -> {output_path}")

    # ファイル取得
    files = download_run.remote(run_id)

    if "error" in files:
        print(f"ERROR: {files['error']}")
        return

    # ローカルに保存
    saved = 0
    for rel_path, content in files.items():
        local_path = output_path / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)
        saved += 1

    print(f"\n完了: {saved} ファイルを {output_path} に保存")

    # レポート表示
    report_path = output_path / "report.md"
    if report_path.exists():
        print(f"\n--- レポートプレビュー ---")
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[:30]:
            print(line.rstrip())
        if len(lines) > 30:
            print(f"... (残り {len(lines) - 30} 行)")
