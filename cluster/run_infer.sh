#!/usr/bin/env bash
#SBATCH --job-name=infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.log
#
# run_infer.sh
#
# Submit infer.py on the UCF-Crime test split (140 anomaly + 150 normal).
# Submit from the scripts directory - splits/ and logs/ are relative to it.
#
#   sbatch run_infer.sh <sampler> <out.jsonl> [extra infer.py args]
#
#   sbatch run_infer.sh motion runs/2b_motion_smoke.jsonl --limit 20
#   sbatch run_infer.sh swift  runs/2b_swift.jsonl
#   sbatch run_infer.sh motion runs/2b_motion.jsonl
#
# Override any default when submitting:
#   MODEL=/data2/bilal/models/qwen3.5-9b sbatch run_infer.sh swift runs/9b_swift.jsonl
#   GPUS=2 sbatch run_infer.sh motion runs/2b_motion.jsonl
#   GPUS=2,3 sbatch --gres=gpu:2 run_infer.sh swift runs/big_swift.jsonl   # split a model
#   PROMPT_ID=2 sbatch run_infer.sh swift runs/2b_swift_p2.jsonl
#
# PROMPT_ID picks the record in prompts.json (system prompt, user prompt,
# response prefix, label words). Default 1, set below. Keep it the same across
# runs you compare, and match the one a LoRA was trained with.
#
# logs/ must exist before submitting (SLURM won't create it): mkdir -p logs
# A timed-out run resumes: resubmit with the same <out.jsonl>.

set -euo pipefail

SAMPLER=${1:?usage: sbatch run_infer.sh <sampler> <out.jsonl> [extra infer.py args]}
OUT=${2:?usage: sbatch run_infer.sh <sampler> <out.jsonl> [extra infer.py args]}
shift 2

PY=${PY:-/data3/bilal/envs/qwen_t5/bin/python}     # ms-swift 4.5.x env, as train_lora.py
MODEL=${MODEL:-/data2/bilal/models/qwen3.5-2b}      # 9b: MODEL=/data2/bilal/models/qwen3.5-9b
ROOT=${ROOT:-/data2/bilal/dataset}                  # prep_ucf.sh's --root
PROMPT_ID=${PROMPT_ID:-1}
GPUS=${GPUS:-0}                                     # which GPU(s); count must match --gres

cd "${SLURM_SUBMIT_DIR:-.}"                         # the scripts directory

# Read before the export, or it just echoes GPUS back. If this is not 0 on a
# single-GPU job, the node does not renumber GPUs per job and the export below
# would take a card we were not allocated - delete it and let SLURM assign.
echo "  SLURM gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
export CUDA_VISIBLE_DEVICES=$GPUS
# torch 2.14 otherwise routes bmm_outer_product through Triton, which needs
# python3.12-dev headers that this node does not have.
export TORCH_DISABLE_NATIVE_JIT=1
# cores for the motion sampler's frame decoding and scoring
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "========================================"
echo "  infer: $SAMPLER -> $OUT"
echo "  Started: $(date)"
echo "  Node:    $(hostname)   GPUs: $GPUS"
echo "  Model:   $MODEL"
# what's already on the GPU(s) - gigabytes used before we load means we landed
# on a card another job is using. nvidia-smi indexes physical GPUs, ignoring
# CUDA_VISIBLE_DEVICES, which is what makes this a real check.
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$GPUS" || true
echo "========================================"

# frame count, token limits and the decord pin are set by infer.py itself
"$PY" infer.py \
    --model "$MODEL" \
    --anomaly splits/test_anomaly.txt \
    --normal  splits/test_normal.txt \
    --video-root "$ROOT" \
    --prompt-id "$PROMPT_ID" \
    --sampler "$SAMPLER" \
    -o "$OUT" "$@"

echo "========================================"
echo "  Finished: $(date)"
echo "  Output:   $OUT"
echo "  Score:    $PY eval.py $OUT"
echo "========================================"