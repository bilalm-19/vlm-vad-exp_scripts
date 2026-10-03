# Data prep

```bash
bash prep_ucf.sh
```

Edit the paths at the top first if the dataset moves.

## What it does

```
path_scanner.py    walk the dirs          -> inventory.txt   (what's on disk)
resolve_split.py   split file + inventory -> real paths      (what's in the split)
prep_ucf.sh        runs both, splits labels by "Normal", checks coverage
```

## Output

| file | lines | read by |
|---|---|---|
| `inventory.txt` | 1900 | `resolve_split.py` |
| `splits/test_anomaly.txt` | 140 | `infer.py` |
| `splits/test_normal.txt` | 150 | `infer.py` |
| `splits/train_anomaly.txt` | 810 | `train_lora.py` |
| `splits/train_normal.txt` | 800 | `train_lora.py` |

Plus `test_all.txt` / `train_all.txt` as intermediates.

**Those counts are the test** — they're UCF's published numbers.

## `loose : 800` is expected

`Anomaly_Train.txt` names its 800 normals under a folder that doesn't exist here:

```
split file asks for:  Training_Normal_Videos_Anomaly/Normal_Videos001_x264.mp4
actually on disk:     /data3/.../Training-Normal-Videos-Part-1/Normal_Videos001_x264.mp4
```

Confirm the folder is fictional:

```bash
ls -d /data2/bilal/dataset/Training_Normal_Videos_Anomaly   # No such file or directory
```

So `resolve_split.py` ignores the folder and matches the filename. All 800
resolve to the right videos — `loose` labels *how* they matched, not a problem.

Safe because filenames are unique across all 1900 videos. Check any time:

```bash
sort inventory.txt | sed 's#.*/##' | uniq -d    # empty = all unique
```

The test split reports no loose matches — it names its normals with no folder,
so there's nothing to disagree with. Loose there would mean something's wrong.

## Other output

`ok: every video on disk is in exactly one split` means the coverage check
passed. Any other step-4 output means a stale inventory (`<`) or a video in both
splits (`>`), and the script halts.

Step 1 is slow — comment it out when nothing on disk has changed.

## Notes

`path_scanner.py` and `resolve_split.py` know nothing about UCF. All the
dataset-specific knowledge is in `prep_ucf.sh`, so a new dataset means a new
`prep_<name>.sh`, not code changes. Both take `--help`.

UCF names its two normal folders differently — `Normal_Videos001` in training,
`Normal_Videos_003` in testing — with one global numbering sequence shared
between them. That's why filenames stay unique and the loose match is safe.