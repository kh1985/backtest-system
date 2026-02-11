# Project: Prism (backtest-system)

## Goal

仮想通貨トレード戦略のバックテスト・分析・自動最適化を行うWebダッシュボードシステム。

- OHLCVデータの読み込み（Binance CSV/ZIP, TradingView CSV）
- GUI上で戦略を組み立てて1回バックテスト実行・分析
- 21種テンプレート × パラメータグリッドサーチ/GA × トレンドレジーム別の自動最適化
- Modal クラウド並列実行（データ取得→最適化→結果DL）
- 最適化結果のJSON保存・読み込み・YAML戦略エクスポート

## Current state

### 完成済み機能
- **データローダー**: Binance CSV/ZIP一括読み込み、TradingView CSV対応、TF自動判定
- **戦略ビルダー**: インジケーター・エントリー条件・決済ルールをGUIで設定（YAML入出力対応）
- **バックテストエンジン**: bar-by-bar シミュレーション（TP/SL/トレーリングストップ/タイムアウト）
- **トレード分析**: 個別トレード詳細チャート・損益分布・勝敗統計・決済タイプ分布
- **自動最適化**: 21種テンプレート × グリッドサーチ/GA × レジーム別。複合スコアでランキング
- **Exit Profiles**: ATR固定/トレーリング/VWAP/BB系 全69パターン + コンパクト3択モード
- **Modal並列実行**: `scripts/modal_optimize.py` でクラウド上バッチ最適化
- **Dual-TF EMAレジーム検出**: 4h+1h EMA合意方式（`analysis/trend.py`）
- **OOS検証**: train 60% / val 20% / test 20% のアウトオブサンプル検証
- **最適化結果JSON読み込み**: `results/` フォルダのJSON → 結果ビュー表示（`OptimizationResultSet.from_json()`）
- **レジーム別ベスト戦略サマリー**: 各レジームの最適戦略カード表示 + YAML一括エクスポート
- **UI**: Streamlit WebUI、ダークテーマ、カスタムCSS、ワークフローガイド

### テンプレート一覧（21種）
| テンプレート | 概要 | Long/Short |
|-------------|------|-----------|
| MA Crossover | SMA fast/slow クロス | Long+Short |
| RSI Reversal | RSI売られすぎ/買われすぎ反発 | Long+Short |
| BB Bounce | ボリンジャーバンド下限/上限タッチ | Long+Short |
| MACD Signal | MACDラインクロス | Long+Short |
| Volume Spike | 出来高急増 + 陰線/陽線反転 | Long+Short |
| Stochastic Reversal | ストキャスティクス K/D クロス | Long+Short |
| VP Pullback | Volume Profile POCへの押し目 | Long |
| VWAP Touch/Sigma | VWAP系エントリー | Long+Short |

### UI名称
- システム名: **Prism**（旧: Backtest System）
- ページタイトル: `Prism`
- サイドバー: `🔷 Prism` + `戦略バックテスト＆最適化ツール`

## Architecture

```
backtest-system/
├── analysis/          # トレンドレジーム検出（MA Cross, ADX, Combined, Dual-TF EMA）
├── config/            # 設定（settings.py: STREAMLIT_PAGE_TITLE等）
├── data/              # データ読み込み（csv_loader, binance_loader, base）
├── engine/            # バックテストエンジン（backtest.py）
├── indicators/        # インジケーター（SMA, EMA, RSI, BB, MACD, ADX, Stoch, VWAP等）
├── metrics/           # パフォーマンス指標計算（calculator.py: BacktestMetrics）
├── optimizer/         # グリッドサーチ・テンプレート・スコアリング・結果（results.py）
├── results/           # 最適化結果JSON保存先
├── scripts/           # Modal実行スクリプト（fetch_data, upload, optimize, download）
├── strategy/          # 戦略定義・ビルダー・YAML例（examples/）
├── ui/
│   ├── app.py         # メインアプリ（Prism）
│   ├── components/    # 共通コンポーネント（chart, styles, metrics_card, trade_table）
│   └── views/         # 各ページ
│       ├── data_loader.py        # データ読み込みページ
│       ├── strategy_builder.py   # 戦略ビルダーページ
│       ├── backtest_runner.py    # バックテスト実行ページ
│       ├── trade_analysis.py     # トレード分析ページ
│       └── optimizer_page.py     # 最適化ページ（設定/結果/読込の3ビュー）
├── inputdata/         # 入力データ置き場
├── sample_data/       # サンプルデータ
└── tests/             # テスト
```

### Key data structures
- `OptimizationEntry`: 1パラメータ組み合わせの結果（template_name, params, trend_regime, metrics, composite_score）
- `OptimizationResultSet`: 最適化結果セット（entries, symbol, execution_tf, htf）。`from_json()` でJSON復元可能
- `BacktestMetrics`: バックテスト指標（trades, win_rate, profit_factor, max_dd, sharpe等）
- `BacktestResult`: バックテスト結果（trades, df, portfolio）

## Tech stack

- Python 3.10+
- Streamlit（WebUI）
- Plotly（チャート）
- pandas / numpy（データ処理）
- ccxt（取引所API）
- PyYAML（戦略設定）
- Modal（クラウド並列実行）

## Coding style

- Python 3.10+
- 関数は小さくテストしやすい単位で
- ログ出力と例外処理を明確に
- ファイル名・関数名・クラス名・変数名は英語
- UI表示テキスト・コメント・説明は日本語

### パフォーマンス最優先ルール（必須）

数値データ・OHLCV処理では**常に最速の実装**を選ぶこと。1分足データ（数十万行）を扱うため、遅いコードは致命的。

**禁止パターン:**
- `for row in df.iterrows()` / `for row in df.itertuples()` でのループ処理
- Python for文による行単位のDataFrame操作
- リスト内包表記で1行ずつ計算してからDataFrameに変換

**必須パターン:**
- pandas のベクトル演算（`df["col"] = df["a"] + df["b"]`）
- numpy の配列演算（`np.where()`, `np.maximum()`, `np.cumsum()` 等）
- pandas の `.shift()`, `.rolling()`, `.expanding()`, `.cummax()`, `.clip()` 等
- boolean indexing（`df[df["rsi"] < 30]`）

**判断基準:** 「Pythonのforループで書けるが、numpy/pandasのベクトル演算でも書ける」場合は、**例外なくベクトル演算を使う**。

## Workflow rules for Claude Code

- **Codex併用**: このプロジェクトは必要に応じてCodexも併用する。ClaudeはCodexによる変更や提案が含まれる前提で扱うこと（不審扱いしない）。
- 複雑な変更は必ず最初にPLAN提示
- 変更は論理的な単位で小さく進める
- ファイル削除や構造変更は事前確認必須
- **ユーザーの明示的な指示なく勝手に作業を進めない**

### 禁止事項（探索フェーズで確定済み — 絶対に再試行しないこと）

| カテゴリ | 禁止パターン | 理由 | 確定Step |
|---------|------------|------|---------|
| テンプレート | Stoch+BB, MACD+BB, RSI+MACD | 全滅（0% PASS） | Step 5 |
| テンプレート | 3重複合（例: RSI+BB+Volume） | 条件過剰→トレード数激減→全滅 | Step 7 |
| テンプレート | VWAP系全6種 | WFA不合格（INJ固有の偽陽性） | Step 13 |
| テンプレート | trend_pullback_long/short | WFA ROBUST率39%だがPnL -7.5%（探索空間405が大） | Step 13 |
| テンプレート | vp_pullback_long | OOS 0%（トレード数不足で全滅） | Step 13 |
| パラメータ | RSI < 29（ロング） | trainで選ばれやすいがOOS全滅の過学習トラップ | Step 8 |
| パラメータ | RSI > 73 / RSI > 75（ショート） | 全レジームで全滅（ロングの29以下と同パターン） | Step 9 |
| パラメータ | BB σ > 2.0 | σ拡張は逆効果（σ=2.0: 48%, σ=2.5: 35%） | Step 7 |
| 方向 | Uptrend × ショート | 理論的に逆張りでリスク大。PASS率高くても除外 | Step 6.5 |
| 方向 | Downtrend × ロング | 方向不一致 | 原則 |
| exit | optimizerにexit選択させる | フォールド間切替が不安定性の主因（ROBUST半減） | Step 12 |

### エラー時の行動指針

- **Modal実行エラー**: ログを確認し原因特定→修正→再実行。3回失敗したらユーザーに報告
- **WFAスクリプトエラー**: `scripts/local_wfa_test.py` のログを確認。データ不足の場合は銘柄をスキップ
- **OOS PASS率が想定外に低い（<10%）**: テンプレートの探索空間サイズを確認（>100 configsなら過学習の疑い）
- **テンプレート追加時**: 上記禁止パターンに該当しないか必ず確認してから実装

### 自動検証ループの起動ルール

ユーザーが「自動で検証して」「ループで回して」「チームで検証」等のニュアンスで指示した場合、**必ず以下を確認してから実行**:

1. **モデル選択を聞く**（AskUserQuestionで）:
   - Opus（判断力重視・コスト高）— 新しいタスク、結果の解釈が必要な場面
   - Sonnet（コスト効率重視）— WFA実行やYAML作成など定型タスクの繰り返し
   - Haiku（最安・最速）— 動作確認やテスト

2. **実行形態を聞く**:
   - Ralph-loop（`scripts/ralph_loop.sh`）— ターミナルで無限ループ。人間不在で回す
   - セッション内チーム — このセッション内でTeam作成して並列実行。人間が監視
   - 単発実行 — 1タスクだけ実行して報告

3. **回数制限を聞く**:
   - `--max-iterations N` or 無制限

勝手にモデルやモードを決めず、ユーザーに選ばせること。

### 戦略探索ワークフロー（自動ガイド）

セッション開始時にロードマップ（`.claude/memory/roadmap.md`）を読んだ場合、以下を自動で報告:
1. **現在のステップ**: 完了済み/進行中のステップ番号と概要
2. **最新ランの結果**: 直近のRun IDとOOS PASS率
3. **次のアクション候補**: ロードマップの「次のステップ」に基づく提案

ユーザーが「next-step」「次のステップ」「次」等と言ったら、`.claude/skills/next-step.md` の手順に従い、詳細な分析→提案→実行まで一連のフローを実行。（VSCodeではスラッシュコマンドが使えないため、キーワードで起動する）

**探索の鉄則（確定済み）:**
- BB系複合テンプレートのみが銘柄横断で機能する
- Stoch+BB, MACD+BB, RSI+MACD は全滅（検証済み、再テスト不要）
- RSI+BB と BB+Volume が唯一の有効パターン
- 新しい複合を試す場合は必ずBBを含める

### メモリ管理

- **作業履歴**: claude-mem（MCP）がバックグラウンドで自動記録。過去の詳細は `search` → `timeline` → `get_observations` で検索可能
- **確定知見** (`MEMORY.md`): 毎セッション自動読み込み。以下のタイミングで**指示を待たず自動更新**する:
  - 新パターンの発見・既存パターンの否定
  - 判定基準や運用方針の変更
  - 環境情報の変化（Modal設定、ファイルパス等）
- **ロードマップ** (`.claude/memory/roadmap.md`): 戦略探索の全体計画・進捗管理。Stepの完了・追加時に**自動更新**する

## Language rules

- 説明・計画・コメントは全て日本語
- ファイル名・ディレクトリ名・関数名・クラス名・コード識別子は英語
- 明示的に求められない限り英語に切り替えない

## 最適化の運用方針（2026-02-09 Step 14完了時点）

### 確定した設定
- **レジーム検出**: Dual-TF EMA（4h+1h合意方式）。`--super-htf 4h`
- **Exit profiles**: exit固定が原則。UT=tp20, DT=tp15
- **OOS**: 必須（train 60% / val 20% / test 20%）
- **最終検証**: WFA（5フォールド Anchored Walk-Forward）。CR>=0.6が合格基準
- **データ**: 30銘柄 × 3年(2023-2026) × 15m/1h/4h

### 過学習防止の原則（Step 8-14で確定）
- **探索空間を小さくする = PASS率改善の鍵**
- exit固定 > exit探索（optimizerにexit選択させない）
- OOS PASSだけでは不十分 → WFAで最終判定
- パラメータ固定（RSI=35, BB=20/2σ, EMA=5/13）が最も安定

### 完全クラウドワークフロー
```
python3 -m modal run scripts/modal_fetch_data.py     # データ取得
python3 -m modal run scripts/modal_optimize.py --exit-profiles atr_compact  # 最適化
python3 -m modal run scripts/modal_download.py       # 結果DL
```
