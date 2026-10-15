r"""
infer.py

Run a VLM over a list of videos and record, per video, the true label, the label
the model chose, and the logprobs of the top candidates for the answer token
(--top-logprobs, 20 by default).

Writes JSONL - computes no metrics (that is done in eval.py)

    TORCH_DISABLE_NATIVE_JIT=1 CUDA_VISIBLE_DEVICES=0 \
        <env>/bin/python infer.py \
        --model <model dir> \
        --anomaly <split>/test_anomaly.txt \
        --normal  <split>/test_normal.txt \
        --video-root <dataset dir> \
        --prompt-id 1 \
        -o <out>.jsonl
 
    TORCH_DISABLE_NATIVE_JIT=1  required under torch 2.14, which otherwise needs
                                python3.12-dev to compile a Triton kernel
    <env>                       must have ms-swift 4.x - not the `qwen` env (3.8)
    --anomaly / --normal        the per-label split files prep_ucf.sh writes
    --video-root                where the videos live. prep_ucf.sh passed --root
                                to path_scanner.py, which stripped that prefix
                                from every path, so it has to go back on here
    --prompt-id                 which record in prompts.json. That record is the
                                only source of the prompt, system prompt,
                                response prefix and the two label tokens, so a
                                run cannot disagree with training
    -o                          appended to, so a killed run resumes
    --adapter                   a LoRA checkpoint from train_lora.py
                                (checkpoint-*/), loaded on top of --model.
                                Its run.json must name the same model and
                                prompt id, or the run stops

One JSON object per video, appended as produced:
 
    path label                   the video and its true label (1 anomaly, 0 normal)
    prompt_id                    which prompts.json record produced this row
    system prompt                its two prompts, stored in full so the row still
                                 means something if prompts.json is edited later
    pred                         the label the model chose, null if neither won
    lp_abnormal lp_normal        logprob of each label at the first token
    top_logprobs                 the top candidates there (--top-logprobs), so
                                 anything can be recomputed without a GPU
    top_token                    the token that actually won
    text                         the raw output, for sanity-checking
    sampler                      the sampler name, always, even "swift"
    sampler_cfg                  the sampler's settings (null for swift)
    frame_idx                    the indices it chose, when it reports them
    run                          what actually ran, read back after loading:
                                 model dir and type, weight dtype, template,
                                 qwen_vl_utils' live video settings, the video
                                 reader, the generation config swift built,
                                 and the LoRA adapter if one was loaded
                                 - not just what was asked for

Nothing derived is stored - no score, no probabilities. They are functions of
the two logprobs, and eval.py computes whatever it needs.

A video whose first token did not offer both labels is still written, with a
"reason" (explaining why it wasnt scored) and no logprobs, so the row count always
matches the split.
 
The response prefix goes on each request, not on the engine - TransformersEngine
accepts the keyword and silently drops it, so the model emits <think> first and
no video scores.

Written against ms-swift 4.5.3 (swift.llm flattened into swift; PtEngine is now
TransformersEngine).
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path


def load_prompt(prompts_file, prompt_id):
    """
    One prompt record from prompts.json - the only place prompt text lives.

    A record carries the prompt, system prompt, response prefix and the two
    label tokens together. Keeping them in one record is what stops inference
    and training drifting onto different answer formats.

    Leading spaces in the labels are significant: Qwen3.5 has ' abnormal' and
    ' normal' as single tokens but no bare 'abnormal' at all, which is why the
    prefix exists. See logprob_scoring_plan.md.
    """
    if not prompts_file.is_file():
        raise SystemExit(f"ERROR: prompts file not found - {prompts_file}")
    records = json.loads(prompts_file.read_text(encoding="utf-8"))
    if prompt_id not in records:
        raise SystemExit(f"ERROR: --prompt-id {prompt_id} not in {prompts_file}. "
                         f"have: {', '.join(sorted(records))}")
    rec = records[prompt_id]
    missing = [k for k in ("prompt", "system", "response_prefix", "answers")
               if k not in rec]
    if missing:
        raise SystemExit(f"ERROR: record {prompt_id} has no {', '.join(missing)}. "
                         f"Logprob scoring needs all four - a record without "
                         f"them is an old tag-format prompt, readable only by "
                         f"parsing the output text.")
    for label in ("Abnormal", "Normal"):
        if label not in rec["answers"]:
            raise SystemExit(f"ERROR: record {prompt_id} has no answers.{label}")
    return rec


def read_lines(path):
    """Video paths from a split file: no blanks, no comments."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def token_text(entry):
    """
    Plain text of one top_logprobs entry, e.g. ' abnormal'.
 
    Needed because the token field is byte-level BPE, where a leading space is
    'Ġ' - comparing that to ' abnormal' would never match. The bytes are
    list(token.encode('utf8')) of the decoded token (infer_engine.py:257), so
    they decode straight to ' abnormal'.
    """
    raw = entry.get("bytes")
    if raw:
        return bytes(raw).decode("utf-8", "replace")
    return str(entry.get("token", "")).replace("Ġ", " ").replace("Ċ", "\n")


def read_labels(steps, labels):
    """
    Logprobs of both labels at the first generated token.
 
    The response prefix ends mid-sentence ("Answer:"), so the very next token is
    the answer - there is nothing to search for. If a label is missing from the
    candidates there, the prefix did not do its job and the video is recorded as
    a failure, rather than quietly scored from some later token.
 
    Matching is exact, so 'normalized' and ' Normal' cannot stand in for
    ' normal'.
 
    Returns ({label: logprob}, winning_token) or (None, None).
    """
    if not steps:
        return None, None
    candidates = steps[0].get("top_logprobs") or []     # ~20 tokens it weighed up
    found = {token_text(c): c["logprob"]
             for c in candidates if token_text(c) in labels}
    if len(found) != len(labels):
        return None, None
    return found, token_text(max(candidates, key=lambda c: c["logprob"]))

def check_single_token(model_dir, labels):
    """
    Each label must be exactly one token in this model's vocabulary.
 
    We read only the first generated token, so a two-token label never matches:
    'Abnormal' tokenises as 'Ab' + 'normal', and the first token holds only
    'Ab'. Every video would fail. Checked here so that costs seconds instead of
    a whole run. Qwen3.5 has ' abnormal' and ' normal' as single tokens but no
    bare 'abnormal' at all - hence the response prefix, which supplies the space.
 
    Returns True when every label is a single token.
    """
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        # the model's own tokeniser; if it won't load, the model won't either
        raise SystemExit(f"could not load tokeniser from {model_dir}: "
                         f"{type(e).__name__}: {e}")
    ok = True
    for label in labels:
        ids = tk.encode(label, add_special_tokens=False)
        note = "ok" if len(ids) == 1 else "NOT ONE TOKEN"
        print(f"  label {label!r:12} -> {len(ids)} token(s) {ids}  {note}")
        ok &= len(ids) == 1
    return ok

def already_done(out_path):
    """
    Paths present in an existing output, so a killed run can be resumed.
 
    Parsed straight, not via read_lines: these are JSON lines, and stripping or
    rewriting characters in them would corrupt the escapes.
    """
    done = set()
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:
                    continue      # a half-written final line from a crash
    return done

def actual_run(engine, cfg, a, adapter):
    """
    What this run really used, read from the objects that do the work rather
    than copied from the arguments:

      model_dir / model_type   what swift loaded (model_info)
      dtype                    the dtype of the loaded weights themselves
      vision                   qwen_vl_utils' module settings - the values
                               fetch_video reads on every call
      video_reader             the backend fetch_video actually picks
      generation               the GenerationConfig swift builds from cfg -
                               temperature 0 becomes do_sample=False (greedy)
      adapter                  the LoRA checkpoint, and whether the loaded
                               model actually carries it (peft_config)
    """
    from qwen_vl_utils import vision_process as vp
    gen = engine._prepare_generation_config(cfg)      # what generate() is given
    return {
        "model": os.path.realpath(a.model),           # what was asked for, resolved
        "model_dir": engine.model_dir,
        "model_type": engine.model_info.model_type,
        "dtype": str(next(engine.model.parameters()).dtype),
        "template": engine.template_type,
        # patch_size x SPATIAL_MERGE_SIZE = pixels per token side
        "patch_size": getattr(getattr(engine.processor, "image_processor", None),
                              "patch_size", None),
        "vision": {k: getattr(vp, k) for k in (
            "FPS", "FPS_MIN_FRAMES", "FPS_MAX_FRAMES", "FRAME_FACTOR",
            "VIDEO_MIN_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "IMAGE_MAX_TOKEN_NUM",
            "SPATIAL_MERGE_SIZE") if hasattr(vp, k)},
        "video_reader": vp.get_video_reader_backend(),
        "generation": {k: getattr(gen, k, None) for k in (
            "do_sample", "max_new_tokens", "num_beams", "repetition_penalty")},
        "top_logprobs": cfg.top_logprobs,
        "adapter": None if adapter is None else {
            **adapter, "loaded": hasattr(engine.model, "peft_config")},
    }


def check_adapter(adapter, model, prompt_id):
    """
    The LoRA checkpoint to load, checked against the run that trained it.

    train_lora.py writes run.json into the run folder, a level or two above the
    checkpoint. An adapter trained on another model or another prompt would
    still load and score - just meaninglessly - so a mismatch stops the run.
    Returns what is recorded in each row: the path and the LoRA settings read
    from the adapter itself.
    """
    path = Path(adapter).expanduser().resolve()
    cfg_file = path / "adapter_config.json"
    if not cfg_file.is_file():
        raise SystemExit(f"--adapter {path}: no adapter_config.json - pass the "
                         f"checkpoint-*/ folder, not the run folder")
    lora = json.loads(cfg_file.read_text(encoding="utf-8"))

    run_json = next((d / "run.json" for d in (path.parent, path.parent.parent)
                     if (d / "run.json").is_file()), None)
    if run_json is None:
        raise SystemExit(f"--adapter {path}: no run.json above it, so the model and "
                         f"prompt it was trained with cannot be checked")
    trained = json.loads(run_json.read_text(encoding="utf-8"))
    t_model = trained["settings"]["model"]
    t_prompt = str(trained["prompt"]["prompt_id"])
    if os.path.realpath(t_model) != os.path.realpath(model):
        raise SystemExit(f"--adapter was trained on {t_model}, not {model}")
    if t_prompt != str(prompt_id):
        raise SystemExit(f"--adapter was trained on prompt {t_prompt}, "
                         f"not --prompt-id {prompt_id}")
    return {"path": str(path), "trained_run": trained["run_name"],
            "r": lora.get("r"), "lora_alpha": lora.get("lora_alpha")}

def existing_setup(out_path):
    """The first row of an existing output, to check a resume matches it."""
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    return json.loads(line)
                except Exception:
                    continue
    return None



def summarise(rows, sampler_name, sampler_calls):
    """
    Print whether this run's output is trustworthy. No metrics - that is
    eval.py. Each line catches a failure that is otherwise silent:
 
      top_token    not the two labels   -> the response prefix is not holding
      leftover     not near 0           -> the model put its mass elsewhere, so
                                           it was not answering the question
      margin       few distinct values  -> the logprobs repeat across videos, so
                                           AUC-ROC will degrade back to ties
      reader calls 0                    -> the sampler override never took, and
                                           the frames are uniform regardless of
                                           what the sampler field says
    """
    scored = [r for r in rows if "lp_abnormal" in r]
    print(f"\n  scored {len(scored)}, failed {len(rows) - len(scored)}")
    if not scored:
        return
 
    tokens = Counter(r["top_token"] for r in scored)
    print("\n  token that won the first step:")
    for tok, n in tokens.most_common(8):
        print(f"    {n:5}  {tok!r}")
    print("    (want essentially all on the two labels)")
 
    # for each video, the probability it gave to neither label.
    # exp() turns each logprob back into a probability; sorting lets the
    # percentiles below be plain index lookups.
    left = sorted(1.0 - math.exp(r["lp_abnormal"]) - math.exp(r["lp_normal"])
                  for r in scored)
    n = len(left)
    print(f"  leftover: median={left[n // 2]:.4f}  "
          f"p95={left[int(0.95 * n)]:.4f}  max={left[-1]:.4f}")
    print("    (probability on neither label - want near 0)")
 
    # for each video, how much more probability went to abnormal than to normal
    g = [r["lp_abnormal"] - r["lp_normal"] for r in scored]
    print(f"  margin: min={min(g):+.2f}  median={statistics.median(g):+.2f}  "
          f"max={max(g):+.2f}  "
          # set() drops duplicates, so this counts videos with an unshared value
          f"distinct={len(set(g))}/{len(g)}")
    print("    (positive = abnormal won; distinct near the total means no ties)")
 
    if sampler_name != "swift":
        read_fail = sum("sampler failed" in r.get("reason", "") for r in rows)
        expected = len(rows) - read_fail
        ok = "ok" if sampler_calls == expected else "MISMATCH"
        print(f"\n  sampler {sampler_name!r}: {sampler_calls} reader calls for "
              f"{expected} readable videos ({read_fail} failed to read)  {ok}")
        if sampler_calls == 0:
            print("    0 calls means the override never took - these frames are"
                  " uniform, not sampled")
 

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="infer.py",
        description="Record the logprobs a VLM puts on each label, per video.")
    ap.add_argument("--model", required=True, help="Model directory")
    ap.add_argument("--adapter",
                    help="LoRA checkpoint folder (checkpoint-*/) from train_lora.py, "
                         "loaded on top of --model")
    ap.add_argument("--anomaly", type=Path, help="Split file of anomaly videos (label 1)")
    ap.add_argument("--normal", type=Path, help="Split file of normal videos (label 0)")
    ap.add_argument("--video-root", type=Path,
                help="Prefix for relative paths in the split files - the "
                        "same --root prep_ucf.sh passed to path_scanner.py. "
                        "Absolute paths in the split are left alone.")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output .jsonl")

    ap.add_argument("--prompts-file", type=Path,
                    default=Path(__file__).resolve().parent / "prompts.json",
                    help="Every prompt lives here, nowhere else")
    ap.add_argument("--prompt-id", required=True,
                    help="Which record in --prompts-file. Supplies the prompt, "
                         "system prompt, response prefix and both label tokens")
    
    ap.add_argument("--sampler", default="swift",
                    help="swift = native uniform sampling, unpatched (default). "
                         "Anything else is looked up in samplers.py.")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--image-max-token-num", type=int, default=1024)
    ap.add_argument("--video-max-token-num", type=int, default=128,
                    help="Max tokens per frame. 128 = qwen_vl_utils' minimum, so "
                         "it is the size actually used; lower is overridden")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--limit", type=int, help="Stop after N videos (smoke test)")
    ap.add_argument("--fresh", action="store_true",
                    help="Overwrite the output instead of resuming it")
    a = ap.parse_args()
 
    rec = load_prompt(a.prompts_file, a.prompt_id)
    adapter = check_adapter(a.adapter, a.model, a.prompt_id) if a.adapter else None
    if not (a.anomaly or a.normal):
        print("ERROR: give --anomaly and/or --normal", file=sys.stderr)
        return 1
    labels = [rec["answers"]["Abnormal"], rec["answers"]["Normal"]]

    if labels[0] == labels[1]:
        print("ERROR: the two labels are identical", file=sys.stderr)
        return 1
 
    # (path, label) in file order; anomaly first so a --limit smoke test sees both
    todo = []
    for split_file, label in ((a.anomaly, 1), (a.normal, 0)):
        if split_file is None:
            continue
        if not split_file.is_file():
            print(f"ERROR: split file not found - {split_file}", file=sys.stderr)
            return 1
        todo += [(p, label) for p in read_lines(split_file)]
    if not todo:
        print("ERROR: no videos in the split files", file=sys.stderr)
        return 1
    
    # the split files may hold paths relative to --root; resolve to what the
    # decoder will actually open. an absolute entry overrides the root, which is
    # how one inventory can span two filesystems.
    def resolve(p):
        return str(a.video_root / p) if a.video_root else p

    # fail now rather than after 290 unreadable videos
    missing = [p for p, _ in todo[:20] if not os.path.exists(resolve(p))]
    if missing:
        print(f"ERROR: {len(missing)} of the first {min(20, len(todo))} videos "
              f"do not exist, e.g.\n         {resolve(missing[0])}\n"
              f"       the split files hold relative paths - pass --video-root "
              f"(prep_ucf.sh used --root)", file=sys.stderr)
        return 1
 
    if a.fresh and a.out.is_file():
        a.out.unlink()
    done = already_done(a.out)
    todo = [(p, l) for p, l in todo if p not in done]
    if a.limit:
        todo = todo[:a.limit]
 
    print(f"  model     : {a.model}")
    if adapter:
        print(f"  adapter   : {adapter['path']}  (r {adapter['r']}, "
              f"alpha {adapter['lora_alpha']})")
    print(f"  videos    : {len(todo)} to do"
        f"{f', {len(done)} already in {a.out}' if done else ''}")
    print(f"  prompt    : {a.prompt_id} ({rec['name']})")
    print(f"  system    : {rec['system']!r}")
    print(f"  user      : {rec['prompt']!r}")
    print(f"  prefix    : {rec['response_prefix']!r}")
    print(f"  labels    : {labels[0]!r} {labels[1]!r}")
    print(f"  frames    : {a.num_frames}   sampler: {a.sampler}")
    if a.video_root:
        print(f"  root      : {a.video_root}")
    if not todo:
        print("  nothing to do")
        return 0
    # after the early exit: this loads the tokeniser, which is not free
    if not check_single_token(a.model, labels):
        print("ERROR: every label must be one token, or no video can be scored. "
              "A leading space is usually what is missing.", file=sys.stderr)
        return 1
 
    # swift reads these at import time, so set them before importing it
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(a.image_max_token_num)
    os.environ["VIDEO_MAX_TOKEN_NUM"] = str(a.video_max_token_num)
    os.environ["FPS_MAX_FRAMES"] = str(a.num_frames)
    # pin the reader so a missing decord cannot fall back to torchvision, which
    # would sample uniformly and look like our sampler silently failing. Set,
    # not setdefault: a value left in the shell would otherwise win unnoticed
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
 
    # patch the frame reader before anything inferences, never lazily - and
    # before the model loads, so a bad name costs seconds not minutes
    samplers = None
    if a.sampler != "swift":
        try:
            import samplers
            samplers.install(a.sampler)
        except ImportError as e:
            print(f"ERROR: could not import samplers.py - {e}\n"
                  "       (needs samplers.py next to this script, plus decord, "
                  "opencv-python-headless and qwen-vl-utils; "
                  "use --sampler swift for native uniform sampling)",
                  file=sys.stderr)
            return 1
        except KeyError:
            known = ", ".join(getattr(samplers, "SAMPLERS", {}))
            print(f"ERROR: unknown sampler {a.sampler!r}. Known: {known or '-'}",
                  file=sys.stderr)
            return 1
        print(f"  sampler {a.sampler!r} installed: {samplers.CONFIG}")

    # qwen_vl_utils re-reads a video with torchvision if decord fails, without
    # saying so. samplers.install() blocks that for sampler runs; block it for
    # swift runs too, so a video decord can't read fails in every run alike and
    # baseline and sampler runs score the same videos. decord is untouched.
    if samplers is None:
        from qwen_vl_utils import vision_process as vp

        def no_fallback(ele):
            raise RuntimeError(f"decord could not read {ele.get('video')} - "
                               f"torchvision fallback blocked")

        vp._read_video_torchvision = no_fallback            # what swift wraps
        vp.VIDEO_READER_BACKENDS["torchvision"] = no_fallback

    from swift import InferRequest, RequestConfig, TransformersEngine
 
    print("\n  loading ...", flush=True)
    engine = TransformersEngine(a.model, max_batch_size=1,
                                adapters=[adapter["path"]] if adapter else None)
    cfg = RequestConfig(max_tokens=a.max_tokens, temperature=0,
                        logprobs=True, top_logprobs=a.top_logprobs)

    run = actual_run(engine, cfg, a, adapter)
    if adapter and not run["adapter"]["loaded"]:
        print("ERROR: --adapter was given but the loaded model carries no LoRA",
              file=sys.stderr)
        return 1
    print(f"  loaded    : {run['model_dir']}  ({run['model_type']}, "
          f"{run['dtype']}, template {run['template']})")
    print(f"  vision    : {run['vision']}")
    print(f"  reader    : {run['video_reader']}")
    v = run["vision"]
    if v.get("VIDEO_MIN_TOKEN_NUM", 0) > v.get("VIDEO_MAX_TOKEN_NUM", 1 << 30):
        # qwen_vl_utils raises max to 1.05 x min, so frames get ~min tokens
        print(f"  WARNING: VIDEO_MIN_TOKEN_NUM {v['VIDEO_MIN_TOKEN_NUM']} > "
              f"VIDEO_MAX_TOKEN_NUM {v['VIDEO_MAX_TOKEN_NUM']} - qwen_vl_utils "
              f"uses the min, so frames get ~{v['VIDEO_MIN_TOKEN_NUM']} tokens, "
              f"not {v['VIDEO_MAX_TOKEN_NUM']}", file=sys.stderr)
    if run["video_reader"] != "decord":
        print(f"  WARNING: video reader is {run['video_reader']!r}, not decord",
              file=sys.stderr)

    # a resume must continue the same run, or the file mixes two setups
    prev = existing_setup(a.out)
    if prev is not None:
        clash = [k for k, now in (("run", run), ("sampler", a.sampler),
                                  ("sampler_cfg", samplers.CONFIG if samplers else None),
                                  ("prompt_id", a.prompt_id))
                 if k in prev and json.dumps(prev[k], sort_keys=True)
                                  != json.dumps(now, sort_keys=True)]
        if clash:
            print(f"ERROR: {a.out} was written with a different "
                  f"{', '.join(clash)} - use a new -o (run_infer.sh: NEW=1), "
                  f"or --fresh to overwrite it",
                  file=sys.stderr)
            return 1
 
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # margin = lp_abnormal - lp_normal, so positive means it leaned abnormal.
    # true is the ground truth from the split file, pred is the model's own
    # pick, and 'x' flags the two disagreeing.
    print(f"\n  {'#':>9} {'margin':>7}  true pred   video")
     
    with open(a.out, "a", encoding="utf-8") as out:
        for i, (path, label) in enumerate(todo, 1):
            row = {"path": path, "label": label, "sampler": a.sampler,
                "sampler_cfg": samplers.CONFIG if samplers else None,
                "run": run,
                "prompt_id": a.prompt_id, "prompt": rec["prompt"],
                "system": rec["system"]}
            try:
                req = InferRequest(
                    messages=[{"role": "system", "content": rec["system"]},
                              {"role": "user", "content": rec["prompt"]}],
                    videos=[resolve(path)],
                    chat_template_kwargs={"response_prefix": rec["response_prefix"],
                                          "enable_thinking": False})
                
                choice = engine.infer([req], cfg)[0].choices[0]
                row["text"] = choice.message.content
 
                # which frames the sampler picked, if it reports them
                if samplers is not None:
                    idx = getattr(samplers, "LAST_INDICES", None)
                    if idx:
                        row["frame_idx"] = list(idx)
 
                lp = getattr(choice, "logprobs", None)
                steps = lp.get("content") if isinstance(lp, dict) else lp
                found, top_token = read_labels(steps or [], labels)
 
                if found is None:
                    # a label was not among the first token's candidates: the
                    # prefix did not hold, or one label fell outside the top-N
                    row["reason"] = "label missing at first token"
                else:
                    row["lp_abnormal"] = found[labels[0]]
                    row["lp_normal"] = found[labels[1]]
                    row["top_token"] = top_token
                    # the model's own choice as a label - what keyword parsing
                    # would have given. null when some third token outranked
                    # both labels, which is still scored: leftover will be high
                    row["pred"] = (1 if top_token == labels[0]
                                   else 0 if top_token == labels[1] else None)
                    row["top_logprobs"] = [
                        {"text": token_text(c), "logprob": c["logprob"]}
                        for c in steps[0]["top_logprobs"]]
            except Exception as e:
                # one unreadable video must not end a 290-video run
                row["reason"] = f"{type(e).__name__}: {e}"
 
            out.write(json.dumps(row) + "\n")
            out.flush()           # survive a SLURM kill with the work so far
            rows.append(row)
            margin = ("%+.2f" % (row["lp_abnormal"] - row["lp_normal"])
                      if "lp_abnormal" in row else "--")
            pred = row.get("pred")
            flag = "x" if pred is not None and pred != label else " "
            print(f"  {f'{i}/{len(todo)}':>9} {margin:>7}  {label:>4} "
                  f"{'-' if pred is None else pred:>4} {flag} "
                  f"{path.split('/')[-1]}", flush=True)
 
    calls = getattr(samplers, "CALLS", {}).get("n", 0) if samplers else 0
    summarise(rows, a.sampler, calls)
    print(f"\n  -> {a.out}")
    return 0 if any("lp_abnormal" in r for r in rows) else 1
 
 
if __name__ == "__main__":
    sys.exit(main())
 