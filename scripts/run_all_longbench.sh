#!/bin/bash
# Run all LongBench evaluations: baseline + SDTP (0.7, 0.8, 0.9)

set -e

TASK=${1:-"hotpotqa"}
MAX_SAMPLES=${2:-200}
MAX_NEW_TOKENS=${3:-128}

echo "=========================================="
echo "[LongBench] Running all configurations"
echo "Task: $TASK"
echo "Max samples: $MAX_SAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "=========================================="
echo ""

# 1. Baseline
echo "[1/4] Running Baseline..."
echo "----------------------------------------"
python3 -m src.evaluation.longbench.run_longbench \
  --task "data/LongBench_data/${TASK}.json" \
  --model checkpoints/qwen2-7b-instruct \
  --mode baseline \
  --do_inference \
  --max_samples $MAX_SAMPLES \
  --max_new_tokens $MAX_NEW_TOKENS \
  --output "results/${TASK}_baseline.json"
echo ""

# 2. SDTP keep_ratio=0.9
echo "[2/4] Running SDTP (keep_ratio=0.9)..."
echo "----------------------------------------"
python3 -m src.evaluation.longbench.run_longbench \
  --task "data/LongBench_data/${TASK}.json" \
  --model checkpoints/qwen2-7b-instruct \
  --pruning_module checkpoints/pruning_module.pt \
  --mode sdtp \
  --do_inference \
  --keep_ratio 0.9 \
  --max_samples $MAX_SAMPLES \
  --max_new_tokens $MAX_NEW_TOKENS \
  --output "results/${TASK}_sdtp_0.9.json"
echo ""

# 3. SDTP keep_ratio=0.8
echo "[3/4] Running SDTP (keep_ratio=0.8)..."
echo "----------------------------------------"
python3 -m src.evaluation.longbench.run_longbench \
  --task "data/LongBench_data/${TASK}.json" \
  --model checkpoints/qwen2-7b-instruct \
  --pruning_module checkpoints/pruning_module.pt \
  --mode sdtp \
  --do_inference \
  --keep_ratio 0.8 \
  --max_samples $MAX_SAMPLES \
  --max_new_tokens $MAX_NEW_TOKENS \
  --output "results/${TASK}_sdtp_0.8.json"
echo ""

# 4. SDTP keep_ratio=0.7
echo "[4/4] Running SDTP (keep_ratio=0.7)..."
echo "----------------------------------------"
python3 -m src.evaluation.longbench.run_longbench \
  --task "data/LongBench_data/${TASK}.json" \
  --model checkpoints/qwen2-7b-instruct \
  --pruning_module checkpoints/pruning_module.pt \
  --mode sdtp \
  --do_inference \
  --keep_ratio 0.7 \
  --max_samples $MAX_SAMPLES \
  --max_new_tokens $MAX_NEW_TOKENS \
  --output "results/${TASK}_sdtp_0.7.json"
echo ""

echo "=========================================="
echo "[OK] All configurations completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/${TASK}_baseline.json"
echo "  - results/${TASK}_sdtp_0.9.json"
echo "  - results/${TASK}_sdtp_0.8.json"
echo "  - results/${TASK}_sdtp_0.7.json"

