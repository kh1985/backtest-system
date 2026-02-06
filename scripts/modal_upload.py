"""
Modal Volume にinputdataをアップロードするスクリプト

使い方:
    modal run scripts/modal_upload.py
    modal run scripts/modal_upload.py --inputdata-dir ./inputdata
"""

import os
from pathlib import Path

import modal

vol = modal.Volume.from_name("prism-data", create_if_missing=True)
app = modal.App("prism-upload")


@app.local_entrypoint()
def main(inputdata_dir: str = "inputdata"):
    """ローカルのinputdata/をModal Volumeにアップロード"""
    inputdata_path = Path(inputdata_dir)
    if not inputdata_path.exists():
        print(f"ERROR: {inputdata_path} が見つかりません")
        return

    csv_files = list(inputdata_path.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: {inputdata_path} にCSVファイルがありません")
        return

    print(f"アップロード対象: {len(csv_files)} ファイル")

    with vol.batch_upload() as batch:
        for f in csv_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
            batch.put_file(str(f), f"/{f.name}")

    print(f"\n完了: {len(csv_files)} ファイルをアップロード")
