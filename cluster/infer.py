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
        -o <out>.jsonl
 
    TORCH_DISABLE_NATIVE_JIT=1  required under torch 2.14, which otherwise needs
                                python3.12-dev to compile a Triton kernel
    <env>                       must have ms-swift 4.x - not the `qwen` env (3.8)
    --anomaly / --normal        the per-label split files prep_ucf.sh writes
    --video-root                where the videos live. prep_ucf.sh passed --root
                                to path_scanner.py, which stripped that prefix
                                from every path, so it has to go back on here
    -o                          appended to, so a killed run resumes

One JSON object per video, appended as produced:
 
    path label                   the video and its true label (1 anomaly, 0 normal)
    system prompt                the two prompts this row was produced with
    pred                         the label the model chose, null if neither won
    lp_abnormal lp_normal        logprob of each label at the first token
    top_logprobs                 the top candidates there (--top-logprobs), so
                                 anything can be recomputed without a GPU
    top_token                    the token that actually won
    text                         the raw output, for sanity-checking
    sampler                      the sampler name, always, even "swift"
    frame_idx                    the indices it chose, when it reports them
    
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


SYSTEM_PROMPT = "You are a video anomaly detector."
USER_PROMPT = ("Classify this surveillance video. "
               "Answer with one word: abnormal or normal.")


# Leading spaces are part of the token. Qwen3.5 has ' abnormal' (33418) and
# ' normal' (4472) as single tokens, but no bare 'abnormal' at all - hence the
# response prefix below, which supplies the space. See logprob_scoring_plan.md.
LABEL_ABNORMAL = " abnormal"
LABEL_NORMAL = " normal"
RESPONSE_PREFIX = "Answer:"

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
        print(f"  (could not check label tokens: {type(e).__name__} - continuing)")
        return True               # no tokeniser is not evidence of a bad label
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
        ok = "ok" if sampler_calls == len(rows) else "MISMATCH"
        print(f"\n  sampler {sampler_name!r}: {sampler_calls} reader calls for "
              f"{len(rows)} videos  {ok}")
        if sampler_calls == 0:
            print("    0 calls means the override never took - these frames are"
                  " uniform, not sampled")
 

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="infer.py",
        description="Record the logprobs a VLM puts on each label, per video.")
    ap.add_argument("--model", required=True, help="Model directory")
    ap.add_argument("--anomaly", type=Path, help="Split file of anomaly videos (label 1)")
    ap.add_argument("--normal", type=Path, help="Split file of normal videos (label 0)")
    ap.add_argument("--video-root", type=Path,
                help="Prefix for relative paths in the split files - the "
                        "same --root prep_ucf.sh passed to path_scanner.py. "
                        "Absolute paths in the split are left alone.")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output .jsonl")
    ap.add_argument("--prompt", default=USER_PROMPT)
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    ap.add_argument("--prompt-file", type=Path,
                    help="Read the user prompt from this file instead of --prompt")
    ap.add_argument("--system-file", type=Path,
                    help="Read the system prompt from this file instead of --system")
    ap.add_argument("--label-abnormal", default=LABEL_ABNORMAL,
                    help="Leading space is significant - it is part of the token")
    ap.add_argument("--label-normal", default=LABEL_NORMAL)
    ap.add_argument("--response-prefix", default=RESPONSE_PREFIX,
                    help="Forced start of the reply. Puts the decision at the "
                         "first generated token and supplies the leading space "
                         "that makes both labels single tokens.")
    ap.add_argument("--sampler", default="swift",
                    help="swift = native uniform sampling, unpatched (default). "
                         "Anything else is looked up in samplers.py.")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--image-max-token-num", type=int, default=1024)
    ap.add_argument("--video-max-token-num", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--limit", type=int, help="Stop after N videos (smoke test)")
    ap.add_argument("--fresh", action="store_true",
                    help="Overwrite the output instead of resuming it")
    a = ap.parse_args()
 
    # a prompt file overrides the default above, so a run is reproducible from it
    for src_path, dest in ((a.prompt_file, "prompt"), (a.system_file, "system")):
        if src_path is None:
            continue
        if not src_path.is_file():
            print(f"ERROR: prompt file not found - {src_path}", file=sys.stderr)
            return 1
        setattr(a, dest, src_path.read_text(encoding="utf-8").strip())
 
    if not (a.anomaly or a.normal):
        print("ERROR: give --anomaly and/or --normal", file=sys.stderr)
        return 1
    labels = [a.label_abnormal, a.label_normal]
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
    if a.video_root:
        print(f"  root      : {a.video_root}")


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
    print(f"  videos    : {len(todo)} to do"
          f"{f', {len(done)} already in {a.out}' if done else ''}")
    print(f"  system    : {a.system!r}")
    print(f"  prompt    : {a.prompt!r}")
    print(f"  prefix    : {a.response_prefix!r}")
    print(f"  frames    : {a.num_frames}   sampler: {a.sampler}")
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
    # would sample uniformly and look like our sampler silently failing
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
 
    # patch the frame reader before anything inferences, never lazily - and
    # before the model loads, so a bad name costs seconds not minutes
    samplers = None
    if a.sampler != "swift":
        try:
            import samplers
            samplers.install(a.sampler)
        except ImportError:
            print("ERROR: --sampler needs samplers.py next to this script "
                  "(use --sampler swift for native uniform sampling)",
                  file=sys.stderr)
            return 1
        except KeyError:
            known = ", ".join(getattr(samplers, "SAMPLERS", {}))
            print(f"ERROR: unknown sampler {a.sampler!r}. Known: {known or '-'}",
                  file=sys.stderr)
            return 1
        print(f"  sampler {a.sampler!r} installed")
 
    from swift import InferRequest, RequestConfig, TransformersEngine
 
    print("\n  loading ...", flush=True)
    engine = TransformersEngine(a.model, max_batch_size=1)
    cfg = RequestConfig(max_tokens=a.max_tokens, temperature=0,
                        logprobs=True, top_logprobs=a.top_logprobs)
 
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # margin = lp_abnormal - lp_normal, so positive means it leaned abnormal.
    # true is the ground truth from the split file, pred is the model's own
    # pick, and 'x' flags the two disagreeing.
    print(f"\n  {'#':>9} {'margin':>7}  true pred   video")
     
    with open(a.out, "a", encoding="utf-8") as out:
        for i, (path, label) in enumerate(todo, 1):
            row = {"path": path, "label": label, "sampler": a.sampler,
                   "system": a.system, "prompt": a.prompt}
            try:
                req = InferRequest(
                    messages=[{"role": "system", "content": a.system},
                              {"role": "user", "content": a.prompt}],
                    videos=[resolve(path)],
                    chat_template_kwargs={"response_prefix": a.response_prefix,
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
 