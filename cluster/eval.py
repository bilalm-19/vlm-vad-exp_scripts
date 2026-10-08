r"""
eval.py

Read the JSONL infer.py wrote and report the metrics. Needs no GPU and no model.

    python eval.py runs/qwen2b_test.jsonl
    python eval.py runs/p1.jsonl runs/p2.jsonl runs/p3.jsonl --combine mean

    --combine mean    average each video's log-odds across the files
    --combine vote    average each video's pred, i.e. a majority vote
    (several files with no --combine: each is scored on its own)

THE PRIMARY SCORE is S(V) = P(abnormal) / (P(abnormal) + P(normal)), computed
as the log-odds lp_abnormal - lp_normal. share() shows why those are the same
thing; both are printed so the equality is visible rather than asserted.

Two further read-outs of the same forward pass are printed for comparison:
P(abnormal) on its own, and the word the model actually said as 1 or 0.
See SCORES.

Per file: AUROC under each read-out with its tied-pair count, the confusion
matrix at pred's operating point, the log-odds spread, and infer.py's validity
checks run over the whole file rather than one run.
"""


import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def share(r):
    """
    S(V) = P(abnormal) / (P(abnormal) + P(normal)), with L = lp_ab - lp_nor:

           P_ab               1                1
        ------------  =  -------------  =  ---------  =  sigmoid(L)
         P_ab + P_nor     1 + P_nor         1 + e^-L
                              ----
                              P_ab

    sigmoid(x) = 1/(1+e^-x) maps the reals onto (0,1) and is strictly
    increasing, so in exact arithmetic S(V) ranks identically to L. In float64
    it saturates to 1.0 past L~37 and stops - which is what reporting both rows
    checks. The right-hand form is computed: one exp rather than three.
    """
    d = r["lp_normal"] - r["lp_abnormal"]
    # a defensive cap: exp overflows near 709.8 in float64. Measured d is only
    # about -4 to +4, so the cap is never reached.
    return 1 / (1 + math.exp(min(d, 700)))


# Four ways to read ONE forward pass. Same logprobs - only the number handed to
# AUROC changes.
#
#   log-odds     lp_ab - lp_nor = log(P_ab/P_nor). only the ratio matters -
#                doubling both probabilities leaves the score unchanged
#   S(V)         P_ab/(P_ab+P_nor) - the log-odds squashed into 0-1, so it must
#                rank identically. a check on the row above, see share()
#   P(abnormal)  lp_ab alone (a logprob, but order is unchanged). high when
#                the clip looks abnormal - and also when the model simply put
#                more of its probability on label words at all
#   pred         the decoded word as 1/0, i.e. the sign of the log-odds - all
#                you get from the output text rather than the logprobs
SCORES = (
    ("log-odds", "<- computed",
     lambda r: r["lp_abnormal"] - r["lp_normal"]),
    ("S(V)", "<- reported; must match the row above",
     share),
    ("P(abnormal)", "ablation: drops the denominator",
     lambda r: r["lp_abnormal"]),
    ("pred", "baseline: the output text, no logprobs",
     lambda r: r.get("pred")),
)


def read_rows(path):
    """Every row of one JSONL, scored or not."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue          # a half-written final line from a crash
    return rows


def auc_and_ties(scored):
    """
    AUC-ROC by pair counting, plus how many pairs were tied.

    Mann-Whitney: over every anomaly/normal pair, 1 when the anomaly scored
    higher, 0.5 when equal. Done the O(n^2) way because 140x150 is nothing and
    it reports the tie count for free, which the rank formula hides.

    scored: list of (score, label). Returns (auc, n_ties, n_pairs).
    """
    anomalies = [score for score, label in scored if label == 1]
    normals = [score for score, label in scored if label == 0]
    if not anomalies or not normals:
        return None, 0, 0
    hits = ties = 0.0
    for anom in anomalies:
        for norm in normals:
            if anom > norm:
                hits += 1               # correct order
            elif anom == norm:
                hits += 0.5             # tie - half credit
                ties += 1
            # anom < norm: wrong order, adds nothing
    pairs = len(anomalies) * len(normals)
    return hits / pairs, int(ties), pairs


def confusion(scored):
    """
    TP/FN/TN/FP at the model's own choice, from pred rather than a threshold.

    scored: list of (pred, label), pred already known to be 0 or 1.
    """
    tp = sum(pred == 1 and true == 1 for pred, true in scored)
    fn = sum(pred == 0 and true == 1 for pred, true in scored)
    tn = sum(pred == 0 and true == 0 for pred, true in scored)
    fp = sum(pred == 1 and true == 0 for pred, true in scored)
    return tp, fn, tn, fp


def report_validity(rows):
    """
    Is this file trustworthy? Each check catches a failure that would otherwise
    leave the metrics looking plausible:

      top_token not the two labels   -> the response prefix was not holding
      labels + stop not near 1       -> something else competed for the answer
      >1 prompt/system/sampler       -> two runs were concatenated by accident

    The first two also appear in infer.py's per-run summary; the third needs a
    whole file. infer.py's sampler call count cannot be rechecked here - it is a
    runtime counter and never reaches the JSONL.
    """
    scored = [r for r in rows if "lp_abnormal" in r]
    failed = [r for r in rows if "lp_abnormal" not in r]
    print(f"  rows {len(rows)}: {len(scored)} scored, {len(failed)} failed")
    for reason, n in Counter(r.get("reason", "?") for r in failed).most_common(5):
        print(f"      {n:4}  {reason}")
    if not scored:
        return

    tokens = Counter(r.get("top_token") for r in scored)
    chose = "  ".join(f"{tok!r} {n}" for tok, n in tokens.most_common(4))
    print(f"  model chose: {chose}")

    # Near 1 says the prompt landed: the model was choosing between the two
    # labels, not heading somewhere else. Also catches mass outside the top-N.
    # Stop counts as an empty answer rather than a third verdict - reported
    # alone too, since that is a judgement call.
    acc, stops = [], []
    for r in scored:
        stop = sum(math.exp(c["logprob"]) for c in r.get("top_logprobs", [])
                   if c["text"].startswith("<|") and c["text"].endswith("|>"))
        stops.append(stop)
        acc.append(math.exp(r["lp_abnormal"]) + math.exp(r["lp_normal"]) + stop)
    acc.sort()
    stops.sort()
    print(f"  labels + stop: median={acc[len(acc) // 2]:.4f}  "
          f"min={acc[0]:.4f}   (want near 1 - the prompt landed)")
    print(f"    of which stop: median={stops[len(stops) // 2]:.4f}  "
          f"max={stops[-1]:.4f}")

    # one run per file, or the metrics below are averaging two configurations
    for field in ("sampler", "prompt", "system"):
        seen = {r.get(field) for r in rows if field in r}
        if len(seen) > 1:
            print(f"  WARNING: {len(seen)} different {field} values in one file")


def report_file(path, rows):
    """Metrics for one run."""
    print(f"\n=== {path} ===")
    report_validity(rows)
    scored = [r for r in rows if "lp_abnormal" in r]
    if not scored:
        return

    # the gap between these rows is the finding - nothing about the model
    # differs between them, only the number handed to AUROC
    print(f"\n  AUROC by read-out ({len(scored)} videos):")
    for name, note, score_of in SCORES:
        usable = [(score_of(r), r["label"]) for r in scored
                  if score_of(r) is not None]
        auc, ties, pairs = auc_and_ties(usable)
        if auc is None:
            print(f"    {name:12}      -   only one class present")
            continue
        # pred is null when a third token outranked both labels, so that row
        # can cover fewer videos - flagged, or the rows look comparable
        fewer = f"  [{len(usable)} only]" if len(usable) != len(scored) else ""
        print(f"    {name:12} {auc:.4f}   {ties:5}/{pairs} tied "
              f"({100 * ties / pairs:4.1f}%)   {note}{fewer}")

    # the confusion matrix belongs to pred only - the one read-out with a fixed
    # operating point. the others are continuous, so there is no cell to count.
    graded = [(r["pred"], r["label"]) for r in scored if r.get("pred") is not None]
    if graded and len({true for _, true in graded}) == 2:   # TNR needs negatives
        tp, fn, tn, fp = confusion(graded)
        # no zero-division: the `if` above requires both classes to be present
        tpr, tnr = tp / (tp + fn), tn / (tn + fp)
        print("\n  at the model's own choice (pred):")
        print(f"    TP {tp}  FN {fn}  TN {tn}  FP {fp}")
        print(f"    TPR {tpr:.4f}  TNR {tnr:.4f}  "
              f"accuracy {(tp + tn) / len(graded):.4f}")
        # equal to AUROC pred by construction: for a two-valued score,
        # pair counting with ties at 0.5 reduces to (TPR+TNR)/2
        print(f"    balanced accuracy {(tpr + tnr) / 2:.4f}  "
              f"= AUROC pred above, as a binary score must")

    # the underlying log-odds. same ordering as share(), so repeated log-odds
    # are also tied share() scores - which is what drags AUROC toward pred.
    # distinct counts near the total mean the score has the resolution to
    # separate videos; the range itself does not cause ties.
    m = [r["lp_abnormal"] - r["lp_normal"] for r in scored]
    print(f"\n  log-odds: min={min(m):+.2f}  median={statistics.median(m):+.2f}"
          f"  max={max(m):+.2f}  {len(set(m))} distinct of {len(m)}")

def combine(files, how):
    """
    One score per video, pooled across files - the ensemble.

    mean: average the log-odds across the files.
    vote: average pred - the fraction of files that said abnormal, i.e. a
          majority vote.

    Only videos present in every file are used, so a half-finished run cannot
    quietly change which videos the ensemble covers.
    """
    per_video = defaultdict(dict)       # video -> {file: score}
    labels = {}
    clashes = set()
    for file_path, rows in files:
        for r in rows:
            if "lp_abnormal" not in r:
                continue
            if how == "vote" and r.get("pred") is None:
                continue
            video = r["path"]
            per_video[video][file_path] = (
                r["lp_abnormal"] - r["lp_normal"] if how == "mean"
                else r["pred"])
            # one video labelled two ways means the runs used different split
            # files - it would score the ensemble against the wrong truth
            if labels.setdefault(video, r["label"]) != r["label"]:
                clashes.add(video)

    # a clashing video has no usable truth, so drop it rather than score it
    # against whichever label happened to be read first
    complete = {video: per_file for video, per_file in per_video.items()
                if len(per_file) == len(files)}
    full = {video: per_file for video, per_file in complete.items()
            if video not in clashes}
    print(f"\n=== combined ({how}) over {len(files)} files ===")
    print(f"  {len(full)} videos scored in all files"
          + (f", {len(per_video) - len(complete)} missing from at least one"
             if len(complete) != len(per_video) else "")
          + (f", {len(complete) - len(full)} dropped for clashing labels"
             if len(complete) != len(full) else ""))
    if clashes:
        print(f"  WARNING: {len(clashes)} videos are labelled differently in "
              f"different files, e.g. {sorted(clashes)[0]} - the runs used "
              f"different split files")
    if not full:
        return

    scored = [(statistics.mean(per_file.values()), labels[video])
              for video, per_file in full.items()]
    auc, ties, pairs = auc_and_ties(scored)
    if auc is None:
        print("  only one class present - AUROC needs both")
        return
    all_scores = [score for score, _ in scored]
    print(f"  AUROC        : {auc:.4f}")
    print(f"    {ties} of {pairs} pairs tied ({100 * ties / pairs:.1f}%)"
          + ("   <- voting rounds each score to 0 or 1, hence the ties"
             if how == "vote" and ties else ""))
    print(f"    {len(set(all_scores))} distinct scores over "
          f"{len(all_scores)} videos")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="eval.py",
        description="Score the JSONL infer.py wrote. No GPU, no model.")
    ap.add_argument("files", nargs="+", type=Path, help="One or more .jsonl runs")
    ap.add_argument("--combine", choices=("mean", "vote"),
                    help="Also pool the files into one score per video: mean "
                         "averages the log-odds, vote averages pred")
    a = ap.parse_args()

    loaded = []
    for path in a.files:
        if not path.is_file():
            print(f"ERROR: not found - {path}", file=sys.stderr)
            return 1
        rows = read_rows(path)
        if not rows:
            print(f"ERROR: no readable rows in {path}", file=sys.stderr)
            return 1
        loaded.append((str(path), rows))

    for path, rows in loaded:
        report_file(path, rows)

    if a.combine:
        if len(loaded) < 2:
            print("\n  (--combine needs at least two files)")
        else:
            # pooling identical runs is not an ensemble, it is a copy
            # skip rows with no prompt field, or a single missing one adds a
            # second distinct value and silently suppresses the warning
            prompts = {r["prompt"] for _, rows in loaded for r in rows
                       if "prompt" in r}
            if len(prompts) == 1:
                print("\n  WARNING: every file used the same prompt - combining "
                      "these is not an ensemble")
            combine(loaded, a.combine)
    return 0


if __name__ == "__main__":
    sys.exit(main())