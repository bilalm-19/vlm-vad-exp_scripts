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
    The output filename is automatically appended with the current date/time.
    If --out is not provided, the filename defaults to:
    output_<date_time>.txt
"""

import argparse # to let the script accept args from the command line
from pathlib import Path # Pythons (more) modern way of working with files and dirs
from datetime import datetime

VIDEO_EXTS_DEFAULT = [".mp4", ".avi", ".mkv", ".mov", ".webm"]

def grab_files(dirs, exts, output_name, recursive = False, root_path = None):
    """
    Write a txt file containing the paths of videos in provided folders
    """

    glob = Path.rglob if recursive else Path.glob # rglob() means recursive glob - searches through the directory and subdirs

    files_only = []

    for directory in dirs:
        files_only.extend(
            p for p in glob(Path(directory), "*")
            if p.is_file() and p.suffix.lower() in exts
        )
        
    with open(output_name, "w") as f:
        for p in files_only:
            if root_path is not None and p.is_relative_to(root_path):
                p = p.relative_to(root_path)
            f.write(str(p) + "\n")

    print(f"Saved as {output_name}")


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
        help="Output filename (default: output_<date_time>.txt)"
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

    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out is None:
        args.out = Path(f"output_{timestamp}.txt")
    else:
        args.out = args.out.with_name(
            f"{args.out.stem}_{timestamp}{args.out.suffix}"
        )

    grab_files(
        args.dirs,
        args.exts,
        args.out,
        args.recursive,
        args.root
    )

    return 0


if __name__ == "__main__":
    main()