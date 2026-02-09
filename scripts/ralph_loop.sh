#!/bin/bash
# Ralph-loop: 自律型検証ループ
#
# 使い方:
#   ./scripts/ralph_loop.sh                      # デフォルト（Opus, 無限ループ）
#   ./scripts/ralph_loop.sh --model sonnet       # Sonnet（安い・速い・定型向き）
#   ./scripts/ralph_loop.sh --model opus         # Opus（高い・賢い・判断向き）
#   ./scripts/ralph_loop.sh --max-iterations 5   # 最大5回で停止
#   ./scripts/ralph_loop.sh --dry-run            # 1回だけ実行して停止
#
# モデル使い分けの目安:
#   opus   → 新戦略の評価、結果の解釈、次タスクの判断が必要な場面（デフォルト）
#   sonnet → 既知タスクの繰り返し実行（WFA回す、YAML作る等）
#
# 注意:
#   --dangerously-skip-permissions を使うため、信頼できる環境でのみ実行
#   Ctrl+C で安全に停止可能（現在のClaudeセッション完了後に終了）

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/agent_logs"
PROMPT_FILE="$PROJECT_DIR/AGENT_PROMPT.md"
PROGRESS_FILE="$PROJECT_DIR/.claude/memory/agent_progress.md"
MAX_ITERATIONS=0  # 0 = 無限
DRY_RUN=false
MODEL="claude-opus-4-6"  # デフォルト: Opus

# 引数パース
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; MAX_ITERATIONS=1; shift ;;
    --model)
      case "$2" in
        opus)   MODEL="claude-opus-4-6" ;;
        sonnet) MODEL="claude-sonnet-4-5-20250929" ;;
        haiku)  MODEL="claude-haiku-4-5-20251001" ;;
        *)      MODEL="$2" ;;  # フルID直接指定も可
      esac
      shift 2 ;;
    --help)
      echo "Usage: $0 [--model opus|sonnet|haiku] [--max-iterations N] [--dry-run]"
      echo ""
      echo "Models:"
      echo "  opus   (default) 判断力重視。新しいタスク・結果解釈向き"
      echo "  sonnet           コスト効率重視。定型タスクの繰り返し向き"
      echo "  haiku            最安・最速。単純なスクリプト実行のみ"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

# 進捗ファイル初期化（なければ作成）
if [ ! -f "$PROGRESS_FILE" ]; then
  cat > "$PROGRESS_FILE" << 'INIT'
# Agent Progress（自律ループ用）

## 状態
- status: idle
- last_iteration: 0
- last_completed_task: none

## 完了タスク

## 残タスク
INIT
fi

# モデル表示名
MODEL_SHORT=$(echo "$MODEL" | sed 's/claude-//' | sed 's/-[0-9].*$//')

echo "=== Ralph Loop 開始 ==="
echo "  Project: $PROJECT_DIR"
echo "  Model:   $MODEL_SHORT ($MODEL)"
echo "  Prompt:  $PROMPT_FILE"
echo "  Logs:    $LOG_DIR/"
echo "  Max iterations: $([ $MAX_ITERATIONS -eq 0 ] && echo '無限' || echo $MAX_ITERATIONS)"
echo ""

ITERATION=0
TRAP_EXIT=false
BACKOFF=30  # レート制限時の初期待機秒数（倍々で増加、最大1時間）

# Ctrl+C で安全に停止
trap 'echo ""; echo "=== 停止シグナル受信。現在のイテレーション完了後に終了します ==="; TRAP_EXIT=true' INT

while true; do
  ITERATION=$((ITERATION + 1))

  # 最大イテレーション数チェック
  if [ $MAX_ITERATIONS -gt 0 ] && [ $ITERATION -gt $MAX_ITERATIONS ]; then
    echo "=== 最大イテレーション数 ($MAX_ITERATIONS) に到達。終了 ==="
    break
  fi

  COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short=6 HEAD 2>/dev/null || echo "nogit")
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOGFILE="$LOG_DIR/agent_${TIMESTAMP}_${COMMIT}_iter${ITERATION}.log"

  echo "--- Iteration $ITERATION [$MODEL_SHORT] (commit: $COMMIT, $(date '+%H:%M:%S')) ---"

  # Claude Code 実行
  cd "$PROJECT_DIR"
  claude --dangerously-skip-permissions \
    -p "$(cat "$PROMPT_FILE")" \
    --model "$MODEL" \
    &> "$LOGFILE" || true

  echo "  完了 → $LOGFILE ($(wc -l < "$LOGFILE") lines)"

  # レート制限検知 → 自動バックオフ
  if grep -qi "rate.limit\|429\|capacity\|overloaded\|too many" "$LOGFILE" 2>/dev/null; then
    BACKOFF=$((BACKOFF * 2))
    if [ $BACKOFF -gt 3600 ]; then BACKOFF=3600; fi
    echo "  ⚠ レート制限検知。${BACKOFF}秒待機..."
    sleep "$BACKOFF"
  else
    BACKOFF=30  # 成功したらリセット
  fi

  # 停止シグナルチェック
  if [ "$TRAP_EXIT" = true ]; then
    echo "=== 安全に停止しました (iteration $ITERATION 完了) ==="
    break
  fi

  # 通常の待機
  sleep 5
done

echo ""
echo "=== Ralph Loop 終了 ==="
echo "  総イテレーション: $ITERATION"
echo "  ログ: $LOG_DIR/"
