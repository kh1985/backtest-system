# /run-app - Streamlitアプリ起動

backtest-system の Streamlit アプリをバックグラウンドで起動する。

## 手順

1. 既存のStreamlitプロセスを確認（`ps aux | grep streamlit`）
2. 既存プロセスがあれば:
   - ユーザーに「既存プロセスがあります。再起動しますか？」と確認
   - 承認されたら既存プロセスを停止
3. 以下のコマンドをバックグラウンドで実行:
   ```
   cd /Users/kenjihachiya/Desktop/work/development/backtest-system && streamlit run app.py
   ```
4. 起動後のURL（通常 http://localhost:8501）を表示

## 注意

- 作業ディレクトリは `/Users/kenjihachiya/Desktop/work/development/backtest-system`
- バックグラウンド実行を使用する
- エラーが出た場合はログを表示
