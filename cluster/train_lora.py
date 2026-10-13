#!/usr/bin/env python3
"""
train_lora.py - LoRA fine-tune a video model on SLURM, from resolved path lists.

Builds the dataset and the sbatch script, then submits. swift sft does the
training - the objective is next-token cross-entropy on the assistant turn.

    python train_lora.py \\
        --videos Abnormal=splits/train_anomaly.txt \\
        --videos Normal=splits/train_normal.txt \\
        --prompt-id 1

    Defaults are the base config of the sweep: rank 16 / alpha 32 / lr 3e-4,
    effective batch 16, 1 epoch, vision frozen, 2 GPUs. Every other cell moves
    one of those, so a bare run is the baseline the rest are compared to.

    --dry-run   build everything and print the sbatch script, don't submit

One run, one directory, so a checkpoint is never separated from what produced it:

    <adapters>/<run_name>/
        train_data.jsonl   the rows trained on, assistant turn included
        run.json           every setting, including the prompt record
        slurm.sh           the exact script submitted
        checkpoint-*/      the adapter

The assistant turn is the response prefix plus the label, e.g. "Answer: abnormal".
The prefix must be in the text: swift applies response_prefix at inference only
(template/base.py:1443), so training without it optimises a different conditional
than infer.py reads. verify_tokens() checks this first.

No validation split - one epoch means one checkpoint, nothing to select between.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

def parse_kv(pairs, flag: str) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        key, sep, val = pair.partition("=")
        if not sep or not val:
            raise SystemExit(f"--{flag} expects KEY=VALUE, got '{pair}'")
        out[key.strip()] = val
    return out


# ── Training data ────────────────────────────────────────────────────────────

def load_videos(lists: dict[str, str], root: Path | None) -> list[tuple[str, Path]]:
    """Read each label's path list from resolve_split.py."""
    videos, seen = [], set()
    for label, file in lists.items():
        path = Path(file)
        if not path.exists():
            raise SystemExit(f"--videos {label}: file not found - {path}\n"
                             f"  Produce it with resolve_split.py.")
        kept = missing = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip().replace("\\", "/")
            if not entry or entry.startswith("#"):
                continue
            p = Path(entry)
            if not p.is_absolute():
                if root is None:
                    raise SystemExit(f"{path} holds relative paths, so --root is needed.")
                p = root / p
            if not p.exists():
                missing += 1
                continue
            p = p.resolve()
            if p in seen:  # the same video under two labels would teach both answers
                print(f"  WARNING: listed under more than one label, skipped: {entry}",
                      file=sys.stderr)
                continue
            seen.add(p)
            videos.append((label, p))
            kept += 1
        print(f"  {label:<10} {kept:>6} videos from {path.name}"
              + (f"   ({missing} listed but not on disk)" if missing else ""))
        if missing:
            print(f"  WARNING: {missing} paths under {label} don't exist - stale list? "
                  f"Re-run path_scanner.py and resolve_split.py.", file=sys.stderr)
    return videos


def load_prompt(prompts_file: Path, prompt_id: str) -> dict:
    """
    One prompt record from prompts.json - the only place prompt text lives.

    A record carries the prompt, system prompt, response prefix and the
    per-label answers. Keeping them in one record is what stops training and
    inference drifting apart, so all four are required - there is no default to
    fall back to, because a default is exactly how the two would diverge.
    """
    if not prompts_file.exists():
        raise SystemExit(f"prompts file not found: {prompts_file}")
    prompts = json.loads(prompts_file.read_text(encoding="utf-8"))
    if prompt_id not in prompts:
        raise SystemExit(f"--prompt-id {prompt_id} not in {prompts_file}. "
                         f"Available: {', '.join(sorted(prompts))}")
    rec = prompts[prompt_id]
    missing = [k for k in ("name", "prompt", "system", "response_prefix", "answers")
               if k not in rec]
    if missing:
        raise SystemExit(f"record {prompt_id} has no {', '.join(missing)}. A "
                         f"record without all of them is an old tag-format "
                         f"prompt, which infer.py cannot read by logprob.")
    for label in ("Abnormal", "Normal"):
        if label not in rec["answers"]:
            raise SystemExit(f"record {prompt_id} has no answers.{label}")
    return rec


def verify_tokens(model_dir, prefix, answers) -> bool:
    """
    Does each label land where infer.py reads it?

    infer.py takes the logprobs of the FIRST generated token, with the prompt
    ending in the prefix. So the label must be one token, and appending it must
    not re-tokenise the prefix. Checked here because the alternative is a run
    that trains cleanly, scores like the base model, and says nothing about why.

    Skipped for the tag-format prompts, which have no prefix and are multi-token
    by design - those are scored by parsing the output text, not by logprob.
    """
    if not prefix:
        print("  (no response prefix - tag-format prompt, skipping token check)")
        return True
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        # the model's own tokeniser; if it won't load, training won't either
        raise SystemExit(f"could not load tokeniser from {model_dir}: "
                         f"{type(e).__name__}: {e}")
    pre = tk.encode(prefix, add_special_tokens=False)
    print(f"  prefix {prefix!r} -> {pre}")
    ok = True
    for label, word in sorted(answers.items()):
        alone = tk.encode(word, add_special_tokens=False)
        joined = tk.encode(prefix + word, add_special_tokens=False)
        good = len(alone) == 1 and joined == pre + alone
        ok &= good
        print(f"  {label:<10} {word!r:12} -> {alone}   with prefix {joined}   "
              f"{'ok' if good else 'MISMATCH'}")
    if not ok:
        print("\nERROR: a label is not a single token after the prefix, so training "
              "would optimise a different distribution than infer.py reads. Fix the "
              "prefix or the answer spelling in prompts.json.", file=sys.stderr)
    return ok


def write_train_jsonl(videos, out: Path, system_prompt, user_prompt, answers,
                      prefix, seed, shuffle):
    """
    The format ms-swift trains from: video + prompt + the answer to learn.

    The response prefix is part of the assistant content, not a separate
    setting: swift applies response_prefix at inference only, so this is the
    only way the trained token sequence matches what infer.py reads.
    """
    unknown = sorted({lab for lab, _ in videos} - set(answers))
    if unknown:
        raise SystemExit(f"No training answer for label(s): {', '.join(unknown)}. "
                         f"The --videos labels must match the prompt record's "
                         f"'answers' keys.")
    rows = [{"messages": [{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt},
                          {"role": "assistant", "content": prefix + answers[lab]}],
             "videos": [str(p)]}
            for lab, p in videos]
    if shuffle:
        random.Random(seed).shuffle(rows)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# ── Batch size ───────────────────────────────────────────────────────────────

def resolve_batch(a) -> str:
    """
    Work out gradient accumulation from the effective batch size.

    Effective batch = per_device_batch_size x grad_accum x num_gpus. Pinning
    grad_accum instead means 2 GPUs and 4 GPUs train with different effective
    batches, so runs that differ only in GPU count are not comparable. Deriving
    it keeps the optimisation identical when the cluster is busier or freer.
    """
    per_step = a.batch_size * a.num_gpus

    if a.grad_accum is not None:
        a.effective_batch = per_step * a.grad_accum
        print("  NOTE: --grad-accum was given, so effective batch changes with "
              "--num-gpus and runs on different GPU counts aren't comparable.",
              file=sys.stderr)
        return describe_batch(a)

    if a.effective_batch % per_step:
        options = [n for n in range(per_step, per_step * 33, per_step)]
        raise SystemExit(
            f"--effective-batch {a.effective_batch} is not divisible by "
            f"{a.batch_size} per device x {a.num_gpus} GPUs = {per_step}.\n"
            f"  Nearest usable values: "
            f"{', '.join(str(o) for o in options if abs(o - a.effective_batch) <= per_step)}")

    a.grad_accum = a.effective_batch // per_step
    return describe_batch(a)


def describe_batch(a) -> str:
    return (f"{a.effective_batch} = {a.batch_size} per device "
            f"x {a.grad_accum} accum x {a.num_gpus} GPU(s)")


# ── SLURM ────────────────────────────────────────────────────────────────────

def build_slurm(a, run_name: str, run_dir: Path, log_path: Path,
                train_file: Path) -> str:
    gpus = ",".join(str(i) for i in range(a.gpu_start, a.gpu_start + a.num_gpus))
    multi = a.num_gpus > 1

    swift = [f"{a.swift_bin} sft",
             f"  --model              {a.model}",
             "  --tuner_type         lora",
             f"  --dataset            {train_file}",
              f"  --output_dir         {run_dir}",
              f"  --num_train_epochs   {a.epochs}",
              f"  --learning_rate      {a.lr}",
              f"  --lora_rank          {a.rank}",
              f"  --lora_alpha         {a.alpha}",
              f"  --lora_dropout       {a.dropout}",
              f"  --target_modules     {a.target_modules}",
              # adapters on the LLM only. the vision features already carry the
              # signal; the decision on top of them is what LoRA is fixing.
              # passed explicitly even though these are swift 4.5.3's defaults,
              # because "vision tower frozen" is a claim the write-up makes.
              f"  --freeze_vit         {str(a.freeze_vit).lower()}",
              "  --freeze_aligner     true",
              "  --freeze_llm         false",
              f"  --per_device_train_batch_size  {a.batch_size}",
              f"  --gradient_accumulation_steps  {a.grad_accum}",
              f"  --max_length         {a.max_length}",
              # swift otherwise puts an empty <think></think> block before every
              # answer in training; infer.py's prompt has none, so the two would differ
              "  --add_non_thinking_prefix  false",
              f"  --torch_dtype        {a.torch_dtype}",
              "  --gradient_checkpointing  true",
              "  --save_strategy      epoch",
              f"  --save_total_limit   {a.save_total_limit}",
              f"  --logging_steps      {a.logging_steps}",
              f"  --warmup_ratio       {a.warmup_ratio}",
              f"  --weight_decay       {a.weight_decay}",
              "  --dataloader_num_workers 4",
              "  --load_from_cache_file true"]
    if multi:
        swift.append(f"  --deepspeed {a.deepspeed}")
    if a.wandb_project:
        swift += ["  --report_to          wandb", f"  --run_name           {run_name}"]
    swift_cmd = " \\\n".join(swift)

    launch = ""
    if multi:
        # A free port per job, so concurrent runs don't collide on the default.
        launch = (f'MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); '
                  f"s.bind(('', 0)); print(s.getsockname()[1]); s.close()\")\n"
                  f"export MASTER_PORT\n"
                  f"echo \"  MASTER_PORT: $MASTER_PORT\"\n\n"
                  f"NPROC_PER_NODE={a.num_gpus} ")

    wandb = ""
    if a.wandb_project:
        wandb = (f'export WANDB_PROJECT="{a.wandb_project}"\n'
                 f'export WANDB_RUN_NAME="{run_name}"\n')

    return f"""#!/usr/bin/env bash
#SBATCH --job-name={run_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --gres=gpu:{a.num_gpus}
#SBATCH --time={a.time}

set -euo pipefail

echo "========================================"
echo "  LoRA training: {run_name}"
echo "  Started: $(date)"
echo "  Node:    $(hostname)"
echo "  GPUs:    {a.num_gpus} (indices {gpus})"
echo "  Model:   {a.model}"
echo "  Epochs {a.epochs} | LR {a.lr} | rank {a.rank} | alpha {a.alpha}"
echo "  Data:    {train_file}"
echo "========================================"

export CUDA_VISIBLE_DEVICES={gpus}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export OMP_NUM_THREADS=8

# torch 2.14 otherwise routes bmm_outer_product through Triton, which needs
# python3.12-dev headers that this node does not have.
export TORCH_DISABLE_NATIVE_JIT=1
# Pin the video reader, so a missing decord cannot fall back to torchvision and
# decode different frames than inference does.
export FORCE_QWENVL_VIDEO_READER=decord

# Video settings - these must match inference or the adapter sees a different
# number of frames at train and test time.
export FPS_MAX_FRAMES={a.num_frames}
export IMAGE_MAX_TOKEN_NUM={a.image_max_token_num}
export VIDEO_MAX_TOKEN_NUM={a.video_max_token_num}
{wandb}
{launch}{swift_cmd}

echo "========================================"
echo "  Finished: $(date)"
echo "  Adapter:  {run_dir}"
echo "========================================"
"""


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_lora.py",
        description="LoRA fine-tune on SLURM from resolved path lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--videos", action="append", required=True, metavar="LABEL=FILE",
                   help="Path list for one label, repeatable. From resolve_split.py.")
    g.add_argument("--root", type=Path, default=Path("/data2/bilal/dataset"),
                   help="Root that relative paths are under")
    g.add_argument("--prompts-file", type=Path,
                   default=Path(__file__).resolve().parent / "prompts.json")
    g.add_argument("--prompt-id", default="1", help="Which prompt to train on")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-shuffle", action="store_true")

    g = p.add_argument_group("model and paths")
    g.add_argument("--model", default="/data2/bilal/models/qwen3.5-2b")
    g.add_argument("--swift-bin",
                   default="/data3/bilal/envs/qwen_t5/bin/swift",
                   help="Must be an env with ms-swift 4.5.x and transformers "
                        "5.2-5.16. qwen3_5 support landed in transformers 5.2, "
                        "so older envs cannot load the model at all")
    g.add_argument("--adapter-base", type=Path, default=Path("/data3/bilal/lora_outputs"))
    g.add_argument("--log-dir", type=Path, default=Path("logs"))
    g.add_argument("--tag", default="", help="Extra text in the run name")

    g = p.add_argument_group("LoRA")
    # The base config of the sweep: rank 16 / alpha 32 / lr 3e-4. Every other
    # cell moves ONE of these, so a bare run is the baseline the rest compare to.
    g.add_argument("--rank", type=int, default=16)
    g.add_argument("--alpha", type=int, default=32)
    g.add_argument("--dropout", default="0.05")
    g.add_argument("--target-modules", default="all-linear")
    g.add_argument("--freeze-vit", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep the vision tower frozen. --no-freeze-vit adapts it "
                        "too: more capacity, but 1400 videos is thin for it")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--lr", default="3e-4")
    g.add_argument("--batch-size", type=int, default=1,
                   help="Per-device batch size")
    g.add_argument("--effective-batch", type=int, default=16,
                   help="Videos per optimiser step. Gradient accumulation is derived "
                        "from this so that changing --num-gpus does not change the "
                        "optimisation. This is the number to keep fixed across runs "
                        "you want to compare")
    g.add_argument("--grad-accum", type=int, default=None,
                   help="Set accumulation steps directly instead of deriving them "
                        "from --effective-batch (effective batch then varies with GPUs)")
    g.add_argument("--max-length", type=int, default=4096)
    g.add_argument("--torch-dtype", default="bfloat16")
    g.add_argument("--warmup-ratio", default="0.05")
    g.add_argument("--weight-decay", default="0.01")
    g.add_argument("--logging-steps", type=int, default=10)
    g.add_argument("--save-total-limit", type=int, default=2)

    g = p.add_argument_group("video (must match inference)")
    g.add_argument("--num-frames", type=int, default=16)
    g.add_argument("--image-max-token-num", type=int, default=1024)
    g.add_argument("--video-max-token-num", type=int, default=128)

    g = p.add_argument_group("cluster")
    g.add_argument("--num-gpus", type=int, default=2)
    g.add_argument("--gpu-start", type=int, default=0)
    g.add_argument("--time", default="12:00:00", help="SLURM time limit")
    g.add_argument("--deepspeed", default="zero2", help="Used when --num-gpus > 1")
    g.add_argument("--wandb-project", default="", help="Enables W&B logging when set")
    g.add_argument("--dry-run", action="store_true",
                   help="Build the run directory and print the SLURM script, don't submit")
    return p


def main() -> int:
    a = build_parser().parse_args()

    if not 1 <= a.num_gpus <= 8:
        raise SystemExit(f"--num-gpus must be 1-8, got {a.num_gpus}")
    batch_desc = resolve_batch(a)
    if not Path(a.model).exists():
        raise SystemExit(f"model not found: {a.model}")
    if not a.dry_run and not Path(a.swift_bin).exists():
        raise SystemExit(f"swift not found: {a.swift_bin}")

    root = a.root.expanduser().resolve() if a.root else None
    print()
    videos = load_videos(parse_kv(a.videos, "videos"), root)
    if not videos:
        raise SystemExit("ERROR: no videos resolved. Empty lists, or wrong --root?")

    # the test split is 290 videos, the train splits ~1610. training on the test
    # clips would contaminate every number measured on them afterwards.
    if len(videos) < 600:
        raise SystemExit(
            f"ERROR: only {len(videos)} videos - these look like the TEST splits. "
            f"Training on them would contaminate the reported results. Pass the "
            f"train lists, or lower the check if this is deliberate.")

    # the prompt record is the single source: prompt, system prompt, response
    # prefix and the per-label answers, so training cannot disagree with
    # inference about the answer format. There is deliberately no flag to
    # override any of them - that is how the two would drift apart.
    rec = load_prompt(a.prompts_file, a.prompt_id)
    prompt_name, prompt_text = rec["name"], rec["prompt"]
    system_prompt, prefix, answers = rec["system"], rec["response_prefix"], rec["answers"]

    print(f"\n  prompt {a.prompt_id}  : {prompt_name}")
    print(f"  prefix    : {prefix!r}")
    if not verify_tokens(a.model, prefix, answers):
        return 1

    counts = Counter(lab for lab, _ in videos)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(a.model).name
    run_name = (f"lora_{model_tag}_r{a.rank}a{a.alpha}_e{a.epochs}_lr{a.lr}"
                f"_eb{a.effective_batch}_p{a.prompt_id}"
                f"{'' if a.freeze_vit else '_vit'}"
                f"{'_' + a.tag if a.tag else ''}_{stamp}")

    run_dir = a.adapter_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # in the run folder, so swift's actual video settings stay with the adapter
    log_path = run_dir.resolve() / "train.log"

    train_file = run_dir / "train_data.jsonl"
    n = write_train_jsonl(videos, train_file, system_prompt, prompt_text,
                          answers, prefix, a.seed, shuffle=not a.no_shuffle)

    slurm = build_slurm(a, run_name, run_dir, log_path, train_file)
    slurm_path = run_dir / "slurm.sh"
    slurm_path.write_text(slurm, encoding="utf-8")

    (run_dir / "run.json").write_text(json.dumps(
        {"run_name": run_name,
         "created": datetime.now().astimezone().isoformat(timespec="seconds"),
         "settings": {k: str(v) for k, v in sorted(vars(a).items())},
         "counts": {"total": n, "by_label": dict(counts)},
         "batch": {"effective": a.effective_batch, "per_device": a.batch_size,
                   "grad_accum": a.grad_accum, "num_gpus": a.num_gpus},
         "prompt": {"prompt_id": a.prompt_id, "prompt_name": prompt_name,
                    "prompt_text": prompt_text, "system_prompt": system_prompt,
                    "response_prefix": prefix, "answers": answers},
         "command": " ".join(sys.argv)},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Run       : {run_name}")
    print(f"  Videos    : {n}  "
          f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})")
    print(f"  Batch     : {batch_desc}")
    print(f"  Vision    : {'frozen' if a.freeze_vit else 'ADAPTED'}")
    if len(counts) == 2:
        hi, lo = sorted(counts.values(), reverse=True)
        if lo and hi / lo > 1.5:
            print(f"  WARNING   : imbalanced ({hi}:{lo}) - the model can do well by "
                  f"always answering the majority label", file=sys.stderr)
    print(f"  Data      -> {train_file}")
    print(f"  Settings  -> {run_dir / 'run.json'}")
    print(f"  SLURM     -> {slurm_path}")

    if a.dry_run:
        print("\n=== DRY RUN - not submitted ===\n")
        print(slurm)
        print(f"=== Submit with:  sbatch {slurm_path} ===")
        return 0

    subprocess.run(["sbatch", str(slurm_path)], check=True)
    print(f"\n  Log       -> {log_path}")
    print(f"  Adapter   -> {run_dir}")
    print(f"\n  Monitor:  tail -f {log_path}")
    print("  Status:   squeue -u $USER")
    return 0


if __name__ == "__main__":
    sys.exit(main())