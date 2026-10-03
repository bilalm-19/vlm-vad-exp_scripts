#!/usr/bin/env bash
#
# prep_ucf.sh
#
# Prepares UCF-Crime: index the disk, resolve both official splits, separate the
# labels.
#
#   bash prep_ucf.sh
#
# Produces:
#   file                        lines  made by                      read by
#   inventory.txt                1900  step 1, path_scanner.py      resolve_split.py
#   splits/test_all.txt           290  step 2, resolve_split.py     step 3
#   splits/train_all.txt         1610  step 2, resolve_split.py     step 3
#   splits/test_anomaly.txt       140  step 3, grep -v Normal       infer.py
#   splits/test_normal.txt        150  step 3, grep    Normal       infer.py
#   splits/train_anomaly.txt      810  step 3, grep -v Normal       train_lora.py
#   splits/train_normal.txt       800  step 3, grep    Normal       train_lora.py
#
# All UCF-specific knowledge lives here - the paths, the split files, and the
# fact that only normal videos have "Normal" in their path. That's what keeps
# path_scanner.py and resolve_split.py generic.
#
# Step 1 walks ~1900 files. Comment it out when nothing on disk has changed.

set -euo pipefail  # stop on error, on unset vars, and on failures mid-pipe

DATA=/data2/bilal/dataset
NORMAL_TRAIN=/data3/bilal/ucf_data_cont   # training normals are on another filesystem
PYTHON=${PYTHON:-python3}                 # e.g. PYTHON=/data2/bilal/envs/qwen/bin/python


# ── 1. Index the disk ────────────────────────────────────────────────────────
# --root makes $DATA paths relative; the /data3 normals stay absolute, so one
# inventory spans both filesystems.
echo "== 1. indexing the disk =="
${PYTHON} path_scanner.py \
  "$DATA"/Anomaly-Videos-Part-{1,2,3,4} \
  "$DATA"/Testing_Normal_Videos_Anomaly \
  "$NORMAL_TRAIN"/Training-Normal-Videos-Part-{1,2} \
  --recursive True --root "$DATA" -o inventory.txt


# ── 2. Resolve the official splits ───────────────────────────────────────────
# The train split reports ~800 "loose" matches. Expected, and correct.
# It names its normals under a folder that isn't on this disk:
#     split file:  Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4
#     on disk:     /data3/.../Training-Normal-Videos-Part-1/Normal_Videos001_x264.mp4
# Verify with: ls -d "$DATA"/Training_Normal_Videos_Anomaly   -> no such directory
# So the folder is ignored and the filename is matched instead. Safe, because all
# 1900 filenames are unique - one filename can only mean one video.
# The test split names its normals with no folder at all, so it reports no loose.
echo
echo "== 2. resolving the official splits =="
${PYTHON} resolve_split.py --split "$DATA"/Anomaly_Test.txt  --paths inventory.txt -o splits/test_all.txt
${PYTHON} resolve_split.py --split "$DATA"/Anomaly_Train.txt --paths inventory.txt -o splits/train_all.txt


# ── 3. Separate the labels ───────────────────────────────────────────────────
# UCF-specific: only normal paths contain "Normal". A grep matching nothing
# exits 1 and halts the script - an empty label file means something's wrong.
echo
echo "== 3. separating labels =="
grep -v Normal splits/test_all.txt  > splits/test_anomaly.txt
grep    Normal splits/test_all.txt  > splits/test_normal.txt
grep -v Normal splits/train_all.txt > splits/train_anomaly.txt
grep    Normal splits/train_all.txt > splits/train_normal.txt


# ── 4. Coverage check ────────────────────────────────────────────────────────
echo
echo "== 4. coverage check =="

# sort groups identical lines, uniq -d prints the repeated ones. Checked first
# because the diff below can't see a video that's duplicated in the inventory
# AND claimed by both splits - both sides would match and the leakage would pass.
dupes=$(sort inventory.txt | uniq -d)

# -z is "is this string empty". Empty, or else print the offenders and quit.
[[ -z "$dupes" ]] || { echo "ERROR: duplicate lines in the inventory:"; echo "$dupes"; exit 1; }

# Everything on disk vs everything the two splits claim - should be identical.
# diff prints any difference and exits non-zero, so set -e halts here:
#     '<'  in the inventory only -> on disk, in no split (stale inventory)
#     '>'  in the splits only    -> claimed by both splits (leakage)
diff <(sort inventory.txt) <(sort splits/test_all.txt splits/train_all.txt)

# Only reached if both checks passed, so seeing this line is the proof.
echo "  ok: every video on disk is in exactly one split"


echo
echo "== counts (expected: 290 = 140+150, 1610 = 810+800) =="
wc -l splits/*.txt
echo
echo "inference : splits/test_anomaly.txt + splits/test_normal.txt"
echo "training  : splits/train_anomaly.txt + splits/train_normal.txt"