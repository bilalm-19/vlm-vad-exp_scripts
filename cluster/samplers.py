r"""
samplers.py

Frame samplers for infer.py. Imported only when --sampler is not "swift".

    infer.py ... --sampler motion

install() replaces qwen_vl_utils' decord reader with _read: an exact copy except
for the line that picks frames. A sampler therefore changes only WHICH frames
Qwen sees, never how many or how they're decoded. --check proves the copy matches.

Three traps this file handles:
  1. swift re-wraps the readers when it loads Qwen, from the module attributes
     (vision_process._read_video_*). install() replaces those attributes as
     well as the VIDEO_READER_BACKENDS entries, or the override is lost.
  2. fetch_video catches reader errors and re-reads with uniform torchvision.
     install() replaces torchvision with _no_fallback, so a failed sampler
     fails the video (infer.py records why) instead of silently going uniform.
  3. Qwen3.5 pairs frames and timestamps them from frames_indices, so a sampler
     must return exactly n sorted, unique frame numbers. _check enforces this.

Adding a sampler: write pick_x(path, start, end, n, fps) -> list[int] and add
it to SAMPLERS. Rows record only the name, so each name = one fixed config.

Test one video without loading Qwen:

    python samplers.py video.mp4 --check --save-dir frames/
"""

import cv2
import decord
import numpy as np
import torch
from qwen_vl_utils import vision_process as vp

CALLS = {"n": 0}          # successful reads; infer.py expects one per video
LAST_INDICES = None       # frames the last read used -> row's frame_idx
_PICK = None              # active sampler, set by install()
_ORIGINAL = None          # qwen_vl_utils' own reader, for --check
_ERROR = None             # why the last read failed, for _no_fallback
CONFIG = None             # settings of the installed sampler -> row's sampler_cfg

# ---------------------------------------------------------------- motion ---

MOTION = dict(
    lag=1,                # compare each frame to the one this many back
    smooth=5,             # moving-average window, in frames
    min_gap_s=2.0,        # min seconds between picks (less for short clips)
    z_thresh=3.0,         # how far above the video's usual level counts as motion
    chunk=32,             # frames decoded at once - speed/memory only
)


def motion_scan(path, start, end, cfg=MOTION):
    """
    Score each frame in [start, end] by mean |frame_t - frame_(t-lag)| on
    blurred grayscale frames at native size. Decodes `chunk` frames at a time,
    so memory stays flat. Returns (idx, scores); the first `lag` frames have
    no score.
    """
    lag = cfg["lag"]
    vr = decord.VideoReader(path)

    idx, scores = [], []    # scored frame numbers, and their scores
    prev = None             # last `lag` frames of the previous chunk

    for c in range(start, end + 1, cfg["chunk"]):
        frames = list(range(c, min(c + cfg["chunk"], end + 1)))   # this chunk

        # decode: (frames, height, width, 3), each pixel an R, G, B value 0-255
        rgb = vr.get_batch(frames).asnumpy()

        # colour -> brightness (motion only needs how much a pixel changed),
        # 5x5 Gaussian blur so pixel noise isn't counted as motion, then int16
        # so differences can go negative
        gray = np.stack([cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0)
                         for f in rgb]).astype(np.int16)

        # prepend the previous chunk's last `lag` frames, so this chunk's first
        # frames have a frame `lag` back to compare with
        if prev is not None:
            gray = np.concatenate([prev, gray])

        # gray[lag:] is each frame, gray[:-lag] the frame `lag` before it, so
        # subtracting compares every frame with its partner in one go. abs so
        # brighter/darker don't cancel; mean over pixels gives one score each.
        # The first `lag` frames in gray have no partner, so there are
        # len(gray) - lag scores, belonging to the last that-many frames.
        if len(gray) > lag:
            scores.extend(np.abs(gray[lag:] - gray[:-lag]).mean(axis=(1, 2)))
            idx.extend(frames[-(len(gray) - lag):])

        prev = gray[-lag:]   # keep only what the next chunk needs

    return np.array(idx, int), np.array(scores)


def smooth(scores, window):
    """
    Moving average over `window` frames. Changes shorter than the window
    (5 frames = ~0.17 s at 30 fps) - glitches, flicker - are flattened; real
    events last seconds, so they stay high and keep their peak. window=1 is off.
    """
    if window <= 1 or len(scores) < window:
        return scores
    # repeat the edge scores outward, so the first and last frames are averaged
    # with copies of themselves rather than with zeros (which would lower them)
    padded = np.pad(scores, (window // 2, (window - 1) // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _uniform(start, end, n):
    """
    n evenly spaced frames across [start, end] - qwen_vl_utils' own formula.
    select() fills from these and returns them when there's no clear motion;
    --check uses them to compare against qwen_vl_utils.
    """
    return torch.linspace(start, end, n).round().long().tolist()


def select(idx, scores, fps, start, end, n, min_gap_s, z_thresh):
    """
    Evenly spaced frames, plus frames on clear motion events. With no clear
    motion this is exactly uniform. Returns n sorted frame numbers.
    """
    # gap = min_gap_s, or half the even spacing if smaller, so short clips
    # still fit n picks
    gap = min(min_gap_s, 0.5 * ((end - start + 1) / fps) / n)
    uniform = _uniform(start, end, n)

    if len(idx) == 0:                           # window too short to score
        return uniform

    # motion = clearly above this video's usual level (every frame scores a
    # little from noise). Median/MAD = usual level/spread, robust to the event
    # itself; 0.6745 puts z in standard deviations. Still camera (MAD 0): any
    # score above the median counts.
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    if mad > 0:
        live = np.flatnonzero(0.6745 * (scores - med) / mad > z_thresh)
    else:
        live = np.flatnonzero(scores > med)

    if len(live) == 0:                          # no clear motion
        return uniform

    # motion frames only, highest first (stable: ties keep time order), each
    # at least `gap` from every pick so they don't all land on one burst
    picks = []
    for j in live[np.argsort(-scores[live], kind="stable")]:
        if all(abs(idx[j] - p) / fps >= gap for p in picks):
            picks.append(int(idx[j]))
        if len(picks) == n:
            break

    # fill the rest with the even frame farthest from any pick, i.e. the
    # middle of the biggest empty stretch
    while len(picks) < n:
        rest = [f for f in uniform if f not in picks]
        picks.append(max(rest, key=lambda f: min(abs(f - p) for p in picks)))

    return sorted(picks)   # Qwen needs frames in time order


def pick_motion(path, start, end, n, fps):
    """The `motion` sampler: score frames by motion, smooth, then even frames
    plus frames on clear motion events."""
    idx, scores = motion_scan(path, start, end)
    scores = smooth(scores, MOTION["smooth"])
    return select(idx, scores, fps, start, end, n,
                  MOTION["min_gap_s"], MOTION["z_thresh"])


# names --sampler accepts -> (function that picks the frames, its settings).
SAMPLERS = {
    "motion": (pick_motion, MOTION),
}


# ---------------------------------------------------------------- reader ---

def _check(idx, n, start, end):
    """Reject picks Qwen would misread: wrong count, unsorted, duplicated, out of window."""
    if len(idx) != n:
        raise ValueError(f"sampler returned {len(idx)} frames, need {n}")
    if any(not isinstance(i, int) for i in idx):
        raise ValueError("sampler returned non-int indices")
    if idx != sorted(set(idx)):
        raise ValueError("sampler indices must be sorted and unique")
    if idx[0] < start or idx[-1] > end:
        raise ValueError(f"sampler index outside window [{start}, {end}]")


def _read(ele):
    """qwen_vl_utils 0.0.14 _read_video_decord, with the frame-picking line -> _PICK."""
    global LAST_INDICES
    LAST_INDICES = None

    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = vp.calculate_video_frame_range(
        ele, total_frames, video_fps)
    nframes = vp.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    idx = _PICK(video_path, start_frame, end_frame, nframes, video_fps)
    _check(idx, nframes, start_frame, end_frame)

    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)   # (frames, colour, h, w)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,          # Qwen timestamps frames from these
        total_num_frames=total_frames,
        video_backend="decord",
    )

    LAST_INDICES = idx
    CALLS["n"] += 1               # last, so only a complete read counts
    return video, video_metadata, sample_fps


def _read_or_record(ele):
    """_read, saving any error so _no_fallback can report it."""
    global _ERROR
    try:
        return _read(ele)
    except Exception as e:
        _ERROR = e
        raise


def _no_fallback(ele):
    """Replaces the torchvision reader, so a failed read fails the video
    instead of being silently re-read with uniform frames."""
    raise RuntimeError(f"sampler failed, uniform fallback blocked - "
                       f"{type(_ERROR).__name__}: {_ERROR}")


def install(name):
    """Point qwen_vl_utils' reader at sampler `name`. KeyError if unknown.
    Must run before Qwen loads."""
    global _PICK, _ORIGINAL, CONFIG
    pick, cfg = SAMPLERS[name]
    if getattr(vp, "_patch", False):
        raise RuntimeError("swift already patched qwen_vl_utils - install() "
                           "must run before the model loads")
    if _ORIGINAL is None:
        _ORIGINAL = vp._read_video_decord
    _PICK = pick
    CONFIG = dict(cfg)            # a copy, so infer.py records what actually ran
    # module attribute = what swift wraps when Qwen loads;
    # dict entry       = what fetch_video calls
    vp._read_video_decord = _read_or_record
    vp.VIDEO_READER_BACKENDS["decord"] = _read_or_record
    vp._read_video_torchvision = _no_fallback
    vp.VIDEO_READER_BACKENDS["torchvision"] = _no_fallback


# ------------------------------------------------------------------- cli ---

def _self_check(path):
    """Our reader with uniform picks must equal qwen_vl_utils' reader exactly."""
    global _PICK
    v0, m0, f0 = _ORIGINAL({"video": path})
    saved, _PICK = _PICK, lambda p, s, e, n, fps: _uniform(s, e, n)
    try:
        v1, m1, f1 = _read({"video": path})
    finally:
        _PICK = saved

    ok = m0 == m1 and f0 == f1 and torch.equal(v0, v1)
    print(f"  check: {'ok' if ok else 'FAILED'} - "
          f"{len(m1['frames_indices'])} frames, reader matches qwen_vl_utils")
    if not ok:
        print(f"    original: {m0['frames_indices']}\n"
              f"    ours    : {m1['frames_indices']}")
    return ok


if __name__ == "__main__":
    import argparse
    import os
    import time

    ap = argparse.ArgumentParser(
        description="Show which frames a sampler picks, without loading Qwen.")
    ap.add_argument("video")
    ap.add_argument("--sampler", default="motion", choices=sorted(SAMPLERS))
    ap.add_argument("--max-frames", type=int, default=16,
                    help="what infer.py's --num-frames sets (FPS_MAX_FRAMES)")
    ap.add_argument("--check", action="store_true",
                    help="also verify the reader against qwen_vl_utils' own")
    ap.add_argument("--save-dir", help="save the chosen frames as JPGs here")
    a = ap.parse_args()

    # cap frames per video (qwen_vl_utils' default is 768). In infer.py swift
    # sets this from FPS_MAX_FRAMES when it loads Qwen; no Qwen here, so set it.
    vp.FPS_MAX_FRAMES = a.max_frames
    install(a.sampler)

    if a.check and not _self_check(a.video):
        raise SystemExit(1)

    t0 = time.perf_counter()
    video, meta, _ = _read({"video": a.video})
    elapsed = time.perf_counter() - t0

    fps, total = meta["fps"], meta["total_num_frames"]
    frames = meta["frames_indices"]
    print(f"  video   : {total} frames at {fps:.1f} fps ({total / fps:.2f} s)")
    print(f"  sampler : {a.sampler}, took {elapsed:.2f} s")
    print(f"  frames  : {frames}")
    print(f"  times s : {[round(f / fps, 2) for f in frames]}")

    if a.save_dir:
        from PIL import Image
        os.makedirs(a.save_dir, exist_ok=True)
        for f, img in zip(frames, video.permute(0, 2, 3, 1).numpy()):
            Image.fromarray(img).save(os.path.join(a.save_dir, f"frame_{f:06d}.jpg"))
        print(f"  saved   : {a.save_dir}")