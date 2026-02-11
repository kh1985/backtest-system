"""
1m足CSVから3m足CSVを生成するリサンプリングスクリプト

pandasのresample機能を使用してOHLCVデータを3分足に変換。

使い方:
    # 単一ファイル変換
    python scripts/resample_to_3m.py --input inputdata/BTCUSDT-1m-20250201-20260130-merged.csv

    # ディレクトリ一括変換
    python scripts/resample_to_3m.py --input-dir inputdata --pattern "*-1m-*.csv"

    # 出力先指定
    python scripts/resample_to_3m.py --input inputdata/BTCUSDT-1m-*.csv --output-dir inputdata/3m
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def resample_1m_to_3m(df: pd.DataFrame) -> pd.DataFrame:
    """
    1m足DataFrameを3m足に変換

    Args:
        df: datetime, open, high, low, close, volume カラムを持つDataFrame

    Returns:
        3m足に変換されたDataFrame
    """
    # datetimeをインデックスに設定
    df_indexed = df.set_index("datetime")

    # 3分足にリサンプル
    resampled = df_indexed.resample("3min", label="left", closed="left").agg({
        "open": "first",    # 3分間の最初の値
        "high": "max",      # 3分間の最大値
        "low": "min",       # 3分間の最小値
        "close": "last",    # 3分間の最後の値
        "volume": "sum",    # 3分間の合計
    })

    # NaNを除去（データが欠けている3分間は削除）
    resampled = resampled.dropna()

    # インデックスをカラムに戻す
    result = resampled.reset_index()

    return result


def convert_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    1つのCSVファイルを1m→3mに変換

    Args:
        input_path: 入力ファイルパス（1m足CSV）
        output_path: 出力ファイルパス（指定なしの場合は自動生成）

    Returns:
        出力ファイルパス
    """
    print(f"[読込] {input_path.name}")

    # CSV読み込み（Binance形式）
    df = pd.read_csv(
        input_path,
        header=None,
        names=["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "count",
               "taker_buy_volume", "taker_buy_quote_volume", "ignore"],
    )

    # open_timeをdatetimeに変換（単位混在に対応）
    import numpy as np
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    # 単位を全てミリ秒に正規化
    ts_ms = np.where(
        ts > 1e15, ts / 1000,       # マイクロ秒 → ミリ秒
        np.where(ts > 1e12, ts,      # ミリ秒そのまま
                 ts * 1000)           # 秒 → ミリ秒
    )
    df["datetime"] = pd.to_datetime(ts_ms, unit="ms")

    # OHLCV列のみ抽出
    ohlcv_df = df[["datetime", "open", "high", "low", "close", "volume"]]

    print(f"  元データ: {len(ohlcv_df)} 行（1m足）")

    # 3m足にリサンプル
    resampled = resample_1m_to_3m(ohlcv_df)

    print(f"  変換後: {len(resampled)} 行（3m足）")

    # 出力ファイル名生成
    if output_path is None:
        # SYMBOL-1m-PERIOD.csv → SYMBOL-3m-PERIOD.csv
        output_name = input_path.name.replace("-1m-", "-3m-")
        output_path = input_path.parent / output_name

    # Binance形式で保存（最低限の列のみ）
    # open_timeはdatetimeから逆算
    output_df = pd.DataFrame()
    output_df["open_time"] = (resampled["datetime"].astype("int64") // 10**6).astype("int64")  # ms
    output_df["open"] = resampled["open"]
    output_df["high"] = resampled["high"]
    output_df["low"] = resampled["low"]
    output_df["close"] = resampled["close"]
    output_df["volume"] = resampled["volume"]

    # 残りの列はダミー（Binanceローダーが期待する12列形式）
    output_df["close_time"] = output_df["open_time"] + 3 * 60 * 1000 - 1  # 3分後 - 1ms
    output_df["quote_volume"] = 0
    output_df["count"] = 0
    output_df["taker_buy_volume"] = 0
    output_df["taker_buy_quote_volume"] = 0
    output_df["ignore"] = 0

    # ヘッダーなしで保存（Binance標準）
    output_df.to_csv(output_path, index=False, header=False)

    print(f"  保存: {output_path.name}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="1m足CSVから3m足CSVを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 単一ファイル変換
  python scripts/resample_to_3m.py --input inputdata/BTCUSDT-1m-20250201-20260130-merged.csv

  # ディレクトリ一括変換
  python scripts/resample_to_3m.py --input-dir inputdata --pattern "*-1m-*.csv"

  # 出力先指定
  python scripts/resample_to_3m.py --input-dir inputdata --pattern "*-1m-*.csv" --output-dir inputdata/3m
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=str,
        help="入力ファイルパス（単一ファイル変換）",
    )
    group.add_argument(
        "--input-dir",
        type=str,
        help="入力ディレクトリパス（一括変換）",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*-1m-*.csv",
        help="入力ディレクトリ内で検索するファイルパターン（デフォルト: *-1m-*.csv）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="出力ディレクトリ（指定なしの場合は入力と同じディレクトリ）",
    )

    args = parser.parse_args()

    # 入力ファイルリスト作成
    input_files: List[Path] = []

    if args.input:
        # 単一ファイル
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"エラー: ファイルが見つかりません: {input_path}")
            return
        input_files.append(input_path)

    elif args.input_dir:
        # ディレクトリ一括
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"エラー: ディレクトリが見つかりません: {input_dir}")
            return

        input_files = sorted(input_dir.glob(args.pattern))

        if not input_files:
            print(f"エラー: パターン '{args.pattern}' に一致するファイルが見つかりません")
            return

    # 出力ディレクトリ
    output_dir: Optional[Path] = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力先: {output_dir}\n")

    print("=" * 60)
    print("  1m足 → 3m足 リサンプリング")
    print("=" * 60)
    print(f"対象ファイル数: {len(input_files)}\n")

    # 変換実行
    for input_path in input_files:
        output_path = None
        if output_dir:
            output_name = input_path.name.replace("-1m-", "-3m-")
            output_path = output_dir / output_name

        try:
            convert_file(input_path, output_path)
        except Exception as e:
            print(f"  エラー: {e}\n")

    print("=" * 60)
    print(f"  完了: {len(input_files)} ファイル変換")
    print("=" * 60)


if __name__ == "__main__":
    main()
