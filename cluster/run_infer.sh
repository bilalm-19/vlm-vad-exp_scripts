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
# Submit from the scripts directory - splits/, runs/ and logs/ are relative to it.
#
#   sbatch run_infer.sh <sampler> [extra infer.py args]
#
#   sbatch run_infer.sh swift
#   sbatch run_infer.sh motion --limit 20                 # smoke test
#   NEW=1 sbatch run_infer.sh swift                       # fresh run, old one kept
#   PROMPT_ID=2 sbatch run_infer.sh swift
#   MODEL=/data2/bilal/models/qwen3.5-9b sbatch run_infer.sh swift
#   GPUS=3,6 MODEL=/data2/bilal/models/qwen3.5-9b sbatch --gres=gpu:2 run_infer.sh swift
#   LORA=r8 ADAPTER=/data3/bilal/lora_outputs/<run>/v0-<time>/checkpoint-<n> \
#       sbatch run_infer.sh swift
#
# The output name is built from the settings, so it always says what ran:
#
#   runs/<size>_p<prompt>_<sampler>[_lora-<LORA>][_smoke]_<YYYYmmdd-HHMM>.jsonl
#   e.g. runs/2b_p2_swift_20261015-1432.jsonl
#
# The time is when the run first started. Resubmitting the same settings
# resumes the newest matching file, running only the videos not in it yet;
# NEW=1 starts a fresh, newly stamped file and leaves the old one alone.
# infer.py refuses to resume a file written by a different model, adapter or
# prompt. --limit adds _smoke and always starts a new file, so a smoke test
# never seeds or resumes a full run. OUT=... overrides the name.
#
# PROMPT_ID picks the record in prompts.json. Default 1. A LoRA must be run
# with the prompt it was trained on - infer.py checks this.
#
# logs/ must exist before submitting (SLURM won't create it): mkdir -p logs

set -euo pipefail

SAMPLER=${1:?usage: sbatch run_infer.sh <sampler> [extra infer.py args]}
shift

PY=${PY:-/data3/bilal/envs/qwen_t5/bin/python}     # ms-swift 4.5.x env, as train_lora.py
MODEL=${MODEL:-/data2/bilal/models/qwen3.5-2b}      # 9b: MODEL=/data2/bilal/models/qwen3.5-9b
ROOT=${ROOT:-/data2/bilal/dataset}                  # prep_ucf.sh's --root
PROMPT_ID=${PROMPT_ID:-1}
GPUS=${GPUS:-0}                                     # which GPU(s); count must match --gres
ADAPTER=${ADAPTER:-}                                # LoRA checkpoint-*/ folder
LORA=${LORA:-}                                      # its short name in the output, e.g. r8
NEW=${NEW:-}                                        # 1 = fresh file, don't resume

cd "${SLURM_SUBMIT_DIR:-.}"                         # the scripts directory

# both or neither: an adapter with no name in the file, or a name with no
# adapter, is exactly the confusion the naming is meant to prevent
if [[ -n "$ADAPTER" && -z "$LORA" ]] || [[ -z "$ADAPTER" && -n "$LORA" ]]; then
    echo "ERROR: set ADAPTER and LORA together" >&2; exit 1
fi

# qwen3.5-2b -> 2b; a folder not ending in a size keeps its full name
SIZE=$(basename "$MODEL")
if [[ "$SIZE" =~ -([0-9.]+[bB])$ ]]; then SIZE=${BASH_REMATCH[1],,}; fi

SMOKE=""
for arg in "$@"; do
    if [[ "$arg" == --limit* ]]; then SMOKE="_smoke"; fi
done

# name = what ran + when it first started. A resubmit with the same settings
# finds the newest such file and resumes it; NEW=1 or a smoke test starts fresh.
# _[0-9]* keeps runs/2b_p1_swift_* from also matching runs/2b_p1_swift_smoke_*
STEM=runs/${SIZE}_p${PROMPT_ID}_${SAMPLER}${LORA:+_lora-$LORA}${SMOKE}
if [[ -z "${OUT:-}" ]]; then
    PREV=""
    if [[ -z "$NEW" && -z "$SMOKE" ]]; then
        # newest by the start time in the name, not by when it was last written
        PREV=$(ls "${STEM}"_[0-9]*.jsonl 2>/dev/null | sort | tail -n 1 || true)
    fi
    if [[ -n "$PREV" ]]; then
        OUT=$PREV
        echo "  resuming $OUT  (NEW=1 for a fresh file)"
    else
        OUT=${STEM}_$(date +%Y%m%d-%H%M).jsonl
    fi
fi

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
echo "  Adapter: ${ADAPTER:-none}"
# what's already on the GPU(s) - gigabytes used before we load means we landed
# on a card another job is using. nvidia-smi indexes physical GPUs, ignoring
# CUDA_VISIBLE_DEVICES, which is what makes this a real check.
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$GPUS" || true
echo "========================================"

ADAPTER_ARGS=()
if [[ -n "$ADAPTER" ]]; then ADAPTER_ARGS=(--adapter "$ADAPTER"); fi

# frame count, token limits and the decord pin are set by infer.py itself
"$PY" infer.py \
    --model "$MODEL" \
    ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
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