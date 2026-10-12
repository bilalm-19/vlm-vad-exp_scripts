r"""
samplers.py

Frame samplers for infer.py (--sampler motion); unused for --sampler swift.
install() swaps in _read: qwen_vl_utils' reader with only the frame picking changed.

    python samplers.py video.mp4 --check --save-dir frames/    # test, no Qwen
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


# ---------------------------------------------------------------- motion ---

MOTION = dict(
    lag=1,                # compare each frame to the one this many back
    smooth=5,             # moving-average window, in frames
    min_gap_s=2.0,        # min seconds between picks (less for short clips)
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
    return np.convolve(scores, np.ones(window) / window, mode="same")


def _uniform(start, end, n):
    """
    n evenly spaced frames across [start, end] - qwen_vl_utils' own formula.
    Used when motion can't supply n frames, and by --check.
    """
    return torch.linspace(start, end, n).round().long().tolist()


def select(idx, scores, fps, start, end, n, min_gap_s):
    """
    The n highest-scoring frames, each at least `gap` seconds apart, topped up
    with evenly spaced frames if needed. Returns n sorted frame numbers.
    """
    # gap = min_gap_s, or half the even spacing if that is smaller:
    # frames / fps = seconds; / n = even spacing; * 0.5 = half of it.
    # Short clips can still fit n picks; long videos keep the min_gap_s cap.
    gap = min(min_gap_s, 0.5 * ((end - start + 1) / fps) / n)
    uniform = _uniform(start, end, n)

    # no motion to rank (window too short to score, or a frozen video)
    if len(idx) == 0 or scores.max() == 0:
        return uniform

    # highest score first (-scores: argsort only sorts low to high; stable:
    # tied scores keep time order, so picks are the same every run). Keep a
    # frame only if it is `gap` from every pick, so picks don't all land on
    # one burst of motion.
    picks = []
    for j in np.argsort(-scores, kind="stable"):
        if all(abs(idx[j] - p) / fps >= gap for p in picks):
            picks.append(int(idx[j]))
        if len(picks) == n:
            break

    # top up with evenly spaced frames if the gap rule left fewer than n.
    # Pass 1 (need = gap/2): only those at least half a gap from every pick.
    # Pass 2 (need = 0): any not yet picked - always reaches n, since the n
    # even frames are all different and each pick rules out at most one.
    for need in (gap / 2, 0):
        for f in uniform:
            if len(picks) == n:
                break
            if f not in picks and all(abs(f - p) / fps >= need for p in picks):
                picks.append(f)

    return sorted(picks)   # Qwen needs frames in time order


def pick_motion(path, start, end, n, fps):
    """The `motion` sampler: score frames by motion, smooth, pick n spaced apart."""
    idx, scores = motion_scan(path, start, end)
    scores = smooth(scores, MOTION["smooth"])
    return select(idx, scores, fps, start, end, n, MOTION["min_gap_s"])


# names --sampler accepts -> the function that picks the frames.
# install() looks the name up here; add a line to add a sampler.
SAMPLERS = {
    "motion": pick_motion,
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
    global _PICK, _ORIGINAL
    pick = SAMPLERS[name]
    if getattr(vp, "_patch", False):
        raise RuntimeError("swift already patched qwen_vl_utils - install() "
                           "must run before the model loads")
    if _ORIGINAL is None:
        _ORIGINAL = vp._read_video_decord
    _PICK = pick
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