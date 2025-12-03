#!/usr/bin/env bash
# retrain_ja.sh — MIDとSFTを日本語データで再学習
# ※ JapaneseInstructデータセットは日本語応答のみにフィルタリング済み
set -euo pipefail

# --- 日本語言語設定 ---
export NANOCHAT_LANG=ja

# --- OOM対策 ---
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

# --- 設定 ---
DEVICE_BATCH_SIZE=16
NUM_ITERATIONS=1000
#NUM_ITERATIONS=200  # クイックテスト用
CACHE_DIR="$HOME/.cache/nanochat"

# ---- 計測開始 ----
T0=$(date +%s)

echo "=== 日本語データで MID + SFT 再学習 ==="
echo "DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}, NUM_ITERATIONS=${NUM_ITERATIONS}"

echo "== 0) 日本語identityファイルをコピー =="
cp data/identity_conversations_ja.jsonl "${CACHE_DIR}/identity_conversations_ja.jsonl"

echo "== 1) MID (日本語) =="
python -m scripts.mid_train \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

echo "== 2) SFT (日本語) =="
python -m scripts.chat_sft \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

# ---- 計測終了 ----
T1=$(date +%s)
ELAPSED=$((T1 - T0))
printf "\n== 完了 ==\nTotal elapsed: %d s (%02d:%02d:%02d)\n" \
  "$ELAPSED" "$((ELAPSED/3600))" "$(((ELAPSED%3600)/60))" "$((ELAPSED%60))"

echo "✅ 再学習完了：Web UI → python -m scripts.chat_web"
