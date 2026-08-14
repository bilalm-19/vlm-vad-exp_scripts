"""
path_scanner.py
 
Walk the directories and write every video path into one file.
 
Just a plain file of what exists in the dirs provided - this is NOT the split file.
 
Usage examples:
    python path_scanner.py /path/to/videos --recursive True
 
    python path_scanner.py /path/to/videos /path/to/more_videos \
        --recursive True --root /path/to --out videos.txt
 
    python path_scanner.py /path/to/videos \
        --recursive False --exts .mp4 .avi
 
Output:
    Defaults to inventory.txt, overwritten on each run.
    The inventory is a rebuilt index of the disk, and later steps refer to it by
    a fixed name, so a changing filename would mean editing those every time.
    Pass --timestamp if you want to keep old copies: inventory_<date_time>.txt
"""
 
import argparse # to let the script accept args from the command line
from pathlib import Path # Pythons (more) modern way of working with files and dirs
import sys # for writing warnings to stderr, and for setting the exit code
from datetime import datetime


VIDEO_EXTS_DEFAULT = [".mp4", ".avi", ".mkv", ".mov", ".webm"]

def norm_exts(exts):
    """
    Accept '.mp4', 'mp4', '*.mp4' or '.MP4' and turn them all into '.mp4',
    so a missing dot doesn't silently match nothing.
    """
    return {"." + e.strip().lstrip("*.").lower() for e in exts}

def grab_files(dirs, exts, output_name, recursive = False, root_path = None):
    """
    Write a txt file containing the paths of videos in provided folders
    """

    glob = Path.rglob if recursive else Path.glob # rglob() means recursive glob - searches through the directory and subdirs

    exts = norm_exts(exts)   

    root_path = root_path.expanduser().resolve() if root_path is not None else None

    files_only = []
    seen = set() # absolute paths already added, so nothing is listed twice
    missing_dirs = 0 # counter of dirs passed that dont exist

    for directory in dirs:

        # make every directory absolute
        # before globbing, so the paths coming out of glob() are absolute too
        # and can be compared against root_path.
        directory = Path(directory).expanduser().resolve()          

        # Path.glob on a directory that doesn't exist returns nothing and raises
        # nothing, so a typo would silently produce an empty inventory.
        if not directory.is_dir():
            print(f"WARNING: not a directory, skipping - {directory}", file=sys.stderr)
            missing_dirs += 1
            continue

        found = 0
        for p in sorted(glob(directory, "*")):                        # ← sorted: same output on any machine
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            p = p.resolve()
            if p in seen:                                             # ← overlapping dirs listed a file twice
                continue
            seen.add(p)
            files_only.append(p)
            found += 1
        print(f"  {directory.name}: {found} videos")

    if not files_only:
        print("ERROR: no videos found. Check the directories and --exts.", file=sys.stderr)
        return 1

    output_name.parent.mkdir(parents=True, exist_ok=True)             # ← -o inv/x.txt no longer crashes

    with open(output_name, "w") as f:
        for p in files_only:
            if root_path is not None and p.is_relative_to(root_path):
                p = p.relative_to(root_path)
            f.write(str(p) + "\n")

    print(f"\nSaved {len(files_only)} paths as {output_name}")
    if missing_dirs:
        print(f"{missing_dirs} directories were missing (see warnings above)", file=sys.stderr)
    return 0

def str_to_bool(value):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise argparse.ArgumentTypeError("Expected True or False")

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="path_scanner.py",
        description="List every video under the given directories, one path per line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python path_scanner.py /data/videos --recursive True\n"
            "  python path_scanner.py /data/videos /data/more_videos "
            "--recursive True --root /data -o inventory/videos.txt\n"
            "  python path_scanner.py /data/videos "
            "--recursive False --exts .mp4 .avi\n"
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        help="Write file paths (of the directories of interest, in the next arg) relative to this folder"
    )

    ap.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="Directories to scan"
    )

    ap.add_argument(
        "-o", "--out",
        type=Path,
        default=Path("inventory.txt"),                                # ← ADD
        help="Output filename (default: inventory.txt, overwritten each run)"
    )

    ap.add_argument(
        "--exts",
        nargs="+",
        default=VIDEO_EXTS_DEFAULT,
        help=f"Extensions to include (default: {' '.join(VIDEO_EXTS_DEFAULT)})"
    )

    ap.add_argument(
        "--recursive",
        type=str_to_bool,
        required=True,
        help="Whether to descend into subdirectories (True or False)"
    )
    ap.add_argument(
        "--timestamp",
        action="store_true",
        help="Append the date/time to the filename instead of overwriting"
    )
    args = ap.parse_args()

    if args.timestamp:                                    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = args.out.with_name(
            f"{args.out.stem}_{timestamp}{args.out.suffix}"
        )

    return grab_files(                                   
        args.dirs,
        args.exts,
        args.out,
        args.recursive,
        args.root
    )


if __name__ == "__main__":
    sys.exit(main())                                    

