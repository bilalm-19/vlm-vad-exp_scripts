"""
resolve_split.py

Takes in the inventory of files on disk + the official split file
Returns the same split, using the respective paths on disk

e.g. split file says   Abuse/Abuse028_x264.mp4
     output line is    Anomaly-Videos-Part-1/Abuse/Abuse028_x264.mp4
                             ^ the Part the split file never tells you

Usage examples:
    python resolve_split.py --split Anomaly_Test.txt --paths inventory.txt \
        -o splits/test_all.txt

    python resolve_split.py --split Anomaly_Train.txt --paths inventory.txt \
        -o splits/train_all.txt

Output:
    One path per line, same form as the inventory entries.
    Prints how many resolved, matched loosely (filename only), or weren't found.

    Labels stay mixed, as in the split file. For UCF only normal paths contain
    "Normal", so grep separates them (-v inverts, > writes to a file):
        grep -v Normal splits/test_all.txt > splits/test_anomaly.txt   # 140
        grep    Normal splits/test_all.txt > splits/test_normal.txt    # 150
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def read_lines(path):
    """Useful lines only: no blanks, no comments, backslashes normalised."""
    for line in open(path, encoding="utf-8"):
        line = line.strip().replace("\\", "/")
        if line and not line.startswith("#"):
            yield line


def resolve_split(split_file, paths_file, out_path):
    inventory = list(read_lines(paths_file))

    # Lookup of KEY = filename, VALUE = the full paths it was found at.
    # That way round because the filename is the only part of a split entry we
    # can trust - its folder is often unusable (missing the Part-N, or naming a
    # folder that isn't on disk). So we look up by name and check folders after.
    # A list of values, since two folders can hold the same filename.
    index = defaultdict(list)
    for line in inventory:
        index[line.split("/")[-1]].append(line)

    resolved, loose, missing = [], [], []

    for entry in read_lines(split_file):
        want = entry.split("/")               # e.g. ["Abuse", "Abuse028_x264.mp4"]
        candidates = index.get(want[-1], [])  # same filename = the only possible matches

        # Compare the LAST N pieces, N = however many the split entry has.
        # Lets a bare filename, a "Class/file.mp4" and a full path all work.
        hits = [c for c in candidates if c.split("/")[-len(want):] == want]

        if not hits and candidates:
            # The folder in the split file isn't on disk, but the filename is
            # (all 800 UCF training normals) - accept it, and record which.
            hits = candidates
            loose.append(entry)

        if not hits:
            missing.append(entry)
        else:
            resolved.append(sorted(hits)[0]) # sorted so a repeat run picks the same one

    print(f"  inventory : {len(inventory)} paths")
    print(f"  resolved  : {len(resolved)}")
    if loose:
        print(f"  loose     : {len(loose)} matched on filename only")
        for l in loose[:3]:
            print(f"      {l}")
    if missing:
        print(f"  MISSING   : {len(missing)} not in the inventory", file=sys.stderr)
        for m in missing[:10]:
            print(f"      {m}", file=sys.stderr)

    if not resolved:
        print("ERROR: nothing resolved. Is the inventory current?", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    print(f"\nSaved {len(resolved)} paths as {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="resolve_split.py",
        description="Turn a split file's entries into paths that exist on disk."
    )
    ap.add_argument(
        "--split",
        type=Path,
        required=True,
        help="The official split file to resolve, e.g. Anomaly_Test.txt"
    )
    ap.add_argument(
        "--paths",
        type=Path,
        required=True,
        help="Inventory of what's on disk, from path_scanner.py"
    )
    ap.add_argument(
        "-o", "--out",
        type=Path,
        required=True,
        help="Output .txt"
    )
    args = ap.parse_args()

    for f in (args.split, args.paths):
        if not f.is_file():
            print(f"ERROR: file not found - {f}", file=sys.stderr)
            return 1

    return resolve_split(args.split, args.paths, args.out)


if __name__ == "__main__":
    sys.exit(main())