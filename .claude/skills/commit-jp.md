# /commit-jp - 日本語コミット

変更内容を分析し、日本語のコミットメッセージでコミットする。

## 手順

1. `git status` と `git diff --stat` で変更内容を確認
2. `git diff` で具体的な変更内容を確認
3. `git log --oneline -5` で直近のコミットスタイルを確認
4. 変更内容に基づいてコミットメッセージを生成:
   - prefix: `feat:` / `fix:` / `refactor:` / `docs:` / `chore:` / `style:`
   - 本文は日本語で簡潔に（1行目は50文字以内目安）
   - 必要に応じて2行目以降に詳細を追記
5. ユーザーにメッセージを提示し、確認を求める
6. 承認されたら:
   - 変更ファイルを `git add` する（.env, credentials等の機密ファイルは除外）
   - `git commit` を実行
   - Co-Authored-By ヘッダーを付与

## コミットメッセージの例

```
feat: フィルタベースのロードビュー + ファイル削除機能

- 銘柄・実行TF・データソースのフィルタUI追加
- 選択ファイルのJSON+CSV一括削除（確認ダイアログ付き）
- plotly_chart重複IDエラー修正

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## 注意

- `.env`, `credentials`, `secrets` 等の機密ファイルは絶対にコミットしない
- `inputdata/`, `results/`, `.DS_Store` 等のデータ・システムファイルは除外
- 変更がない場合は「コミットする変更がありません」と報告
- push はしない（ユーザーが明示的に求めた場合のみ）
