#!/usr/bin/env bash
# speedrun_karaage.sh — 自分の文章で学習するスクリプト (DGX Spark 単GPU版)
# mid_trainからやり直して、カスタムスタイルを学習
set -euo pipefail

# ===== ユーザー設定 =====
DEVICE_BATCH_SIZE=16
NUM_ITERATIONS=1000
CACHE_DIR="$HOME/.cache/nanochat"
# ========================

# --- 日本語言語設定 ---
export NANOCHAT_LANG=ja

# --- カスタムデータ設定 ---
export NANOCHAT_EXTRA_DATA="${CACHE_DIR}/karaage_conversations.jsonl"
# identityをスキップ（カスタムスタイルを優先）
export NANOCHAT_SKIP_IDENTITY=1

# --- 実行環境・OOM対策 ---
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

# ---- 計測開始 ----
T0=$(date +%s)

echo "=== nanochat カスタムスタイル学習 (single GPU on DGX Spark) ==="
echo "DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}, NUM_ITERATIONS=${NUM_ITERATIONS}"
echo "NANOCHAT_SKIP_IDENTITY=${NANOCHAT_SKIP_IDENTITY}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
PY

echo "== 1) カスタムデータ準備 =="
echo "Converting blog posts to JSONL..."
python -m scripts.convert_blog_to_jsonl --input ./information-hub --output ./data/karaage_conversations.jsonl
cp data/karaage_conversations.jsonl "${NANOCHAT_EXTRA_DATA}"
echo "カスタムデータ: $(wc -l < data/karaage_conversations.jsonl) 会話"

echo "== 2) MID (カスタムスタイル) =="
python -m scripts.mid_train \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

echo "== 3) SFT (カスタムスタイル) =="
python -m scripts.chat_sft \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

# ---- 計測終了＆表示 ----
T1=$(date +%s)
ELAPSED=$((T1 - T0))
printf "\n== SUMMARY ==\nTotal elapsed: %d s (%02d:%02d:%02d)\n" \
  "$ELAPSED" "$((ELAPSED/3600))" "$(((ELAPSED%3600)/60))" "$((ELAPSED%60))"

echo "✅ カスタムスタイル学習完了！"
echo "Web UI を起動: python -m scripts.chat_web"
echo "CLI で試す: python -m scripts.chat_cli -p 'Raspberry Piについて教えて'"
