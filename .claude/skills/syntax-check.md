# /syntax-check - 変更ファイルの構文チェック

Git で変更されたPythonファイルの構文チェックを行う。

## 手順

1. `git diff --name-only` と `git diff --cached --name-only` で変更済み `.py` ファイルを取得する
2. 各ファイルに対して `python -m py_compile <ファイルパス>` を実行する
3. エラーがあった場合:
   - エラーメッセージの該当行番号と内容を表示
   - 可能であれば修正案を提示（ただし自動修正はしない）
4. 全ファイルがパスした場合: 「✅ N件パス」と簡潔に報告

## 注意

- 作業ディレクトリは `/Users/kenjihachiya/Desktop/work/development/backtest-system`
- 変更ファイルがない場合は「変更ファイルなし」と報告
- `.py` 以外のファイルはスキップ
- 出力は日本語で簡潔に
