# /add-template - 新しいトレード戦略テンプレート追加

新しいトレード戦略テンプレートをシステムに追加する。

## 手順

1. ユーザーにヒアリング:
   - テンプレート名（英語、snake_case）
   - 方向: long / short
   - 使用するインジケーター（RSI, BB, MA等）
   - エントリー条件の概要
   - イグジット条件の概要
   - パラメータとそのグリッドサーチ範囲

2. `optimizer/templates.py` を読み込んで既存テンプレートの構造を確認

3. 新テンプレートを追加:
   - `StrategyTemplate` のインスタンスを作成
   - `BUILTIN_TEMPLATES` dict に追加
   - パラメータのデフォルトレンジを設定

4. 必要に応じて `engine/` 配下にエントリー/イグジットロジックを追加:
   - `engine/strategies/` に新ファイル or 既存ファイルに追加
   - `engine/backtest.py` から呼び出せるようにする

5. 構文チェック（`python -m py_compile`）を実行

6. 変更内容のサマリーを表示

## 注意

- 作業ディレクトリは `/Users/kenjihachiya/Desktop/work/development/backtest-system`
- 既存テンプレートの命名規則・構造に合わせる
- テスト可能な状態になるまで実装する
- 出力は日本語で簡潔に
