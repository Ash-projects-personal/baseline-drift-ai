#!/usr/bin/env python3
"""Camera pipeline for Baseline Drift AI (CSCE 5280), implementing exactly the method
in the submitted Research Strategy (Proposal #2):

- MediaPipe BlazePose landmarks (mp.solutions.pose, model_complexity=1)
- Reject landmarks with visibility < 0.5 (proposal: confidence < 0.5 rejected)
- Smooth landmark trajectories over 5 frames (moving average)
- Features per walking segment: gait speed and step-time variability
- A segment is usable only if >= 6 steps are visible

Input: a directory of PNG frames (one UR Fall Detection Dataset RGB sequence) or a
video file. URFD RGB was recorded at 25 fps (Kwolek & Kepski 2014); fps is taken from
--fps (default 25) and recorded in the output.

Scale note: URFD has no camera calibration and no known scene distances, so gait
speed is reported in two honest units: pixels/second and body-heights/second, where
body height is the subject's own pixel height (shoulder-midpoint to ankle-midpoint
distance, median over usable frames). No metres/second figure is claimed, because no
timed-walk ground truth or calibration exists in this data.

Step detection: heel-strike proxy = peaks of the inter-ankle horizontal distance
(normalized by body height), found with scipy find_peaks (min separation 0.3 s,
prominence 0.05 body heights). Step times are the intervals between consecutive
peaks; step-time variability is the coefficient of variation (SD / mean) of those
intervals. Gait speed is the median hip-centre displacement per second over frames
where both ankles are visible, excluding frames with no usable pose.

Output: one JSON per sequence plus an aggregate JSON.
"""
import argparse
import json
import math
import os
import sys
import warnings

import cv2
import numpy as np


def smooth_5(x):
    """5-frame moving average, edge-padded (proposal: smoothing over five frames)."""
    if len(x) < 5:
        return x.copy()
    k = np.ones(5) / 5.0
    padded = np.pad(x, (2, 2), mode="edge")
    return np.convolve(padded, k, mode="valid")


def load_frames(path):
    """Return list of (name, BGR image). Directory of PNGs (sorted) or a video file."""
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        out = []
        for n in names:
            img = cv2.imread(os.path.join(path, n))
            if img is not None:
                out.append((n, img))
        return out
    cap = cv2.VideoCapture(path)
    out = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append((f"frame_{i:06d}", img))
        i += 1
    cap.release()
    return out


def process_sequence(frames, fps, name):
    import mediapipe as mp
    from scipy.signal import find_peaks

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    L = mp.solutions.pose.PoseLandmark

    n = len(frames)
    # Per-frame raw landmark arrays (33 x 3: x, y, visibility), NaN where absent.
    raw = np.full((n, 33, 3), np.nan)
    detected = np.zeros(n, dtype=bool)
    for i, (_, img) in enumerate(frames):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            continue
        detected[i] = True
        for j, lm in enumerate(res.pose_landmarks.landmark):
            raw[i, j] = (lm.x, lm.y, lm.visibility)
    pose.close()

    # Smoothing over 5 frames (x and y only; visibility not smoothed).
    sm = np.full_like(raw, np.nan)
    for j in range(33):
        for c in (0, 1):
            col = raw[:, j, c]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                continue
            filled = np.interp(np.arange(n), np.flatnonzero(ok), col[ok])
            sm[:, j, c] = smooth_5(filled)
    sm[:, :, 2] = raw[:, :, 2]

    def xy(idx):
        return sm[:, idx, 0], sm[:, idx, 1]

    vis = sm[:, :, 2]  # visibility per landmark per frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ankle_vis = np.nanmin(
            np.stack([vis[:, L.LEFT_ANKLE], vis[:, L.RIGHT_ANKLE],
                      vis[:, L.LEFT_FOOT_INDEX], vis[:, L.RIGHT_FOOT_INDEX]]), axis=0)
    # A frame is usable when a pose was detected and all ankle/foot landmarks are
    # at visibility >= 0.5 (proposal rejection rule).
    usable = detected & (ankle_vis >= 0.5)

    # Body height in pixels per frame: shoulder midpoint to ankle midpoint.
    sh_x = (sm[:, L.LEFT_SHOULDER, 0] + sm[:, L.RIGHT_SHOULDER, 0]) / 2
    sh_y = (sm[:, L.LEFT_SHOULDER, 1] + sm[:, L.RIGHT_SHOULDER, 1]) / 2
    an_x = (sm[:, L.LEFT_ANKLE, 0] + sm[:, L.RIGHT_ANKLE, 0]) / 2
    an_y = (sm[:, L.LEFT_ANKLE, 1] + sm[:, L.RIGHT_ANKLE, 1]) / 2
    h_img, w_img = frames[0][1].shape[:2]
    body_h_px = np.sqrt(((sh_x - an_x) * w_img) ** 2 + ((sh_y - an_y) * h_img) ** 2)
    body_h_med = float(np.nanmedian(body_h_px[usable])) if usable.any() else float("nan")

    # Hip-centre trajectory in pixels, and per-frame speed.
    hip_x = (sm[:, L.LEFT_HIP, 0] + sm[:, L.RIGHT_HIP, 0]) / 2 * w_img
    hip_y = (sm[:, L.LEFT_HIP, 1] + sm[:, L.RIGHT_HIP, 1]) / 2 * h_img
    disp = np.sqrt(np.diff(hip_x) ** 2 + np.diff(hip_y) ** 2)  # px per frame
    speed_px_s = disp * fps
    speed_bh_s = speed_px_s / body_h_med if body_h_med and not math.isnan(body_h_med) else np.full_like(speed_px_s, np.nan)

    # Step detection: peaks of the 2D inter-ankle distance (normalised by body
    # height). 2D, not horizontal-only, because these clips are mostly
    # front-facing walking, where the step signal is in the vertical direction.
    la_x, la_y = xy(L.LEFT_ANKLE)
    ra_x, ra_y = xy(L.RIGHT_ANKLE)
    inter = np.sqrt(((la_x - ra_x) * w_img) ** 2 + ((la_y - ra_y) * h_img) ** 2) / body_h_med
    inter[~usable] = np.nan
    ok = ~np.isnan(inter)
    if ok.sum() > 10:
        filled = np.interp(np.arange(n), np.flatnonzero(ok), inter[ok])
        peaks, _ = find_peaks(filled, distance=int(0.3 * fps), prominence=0.05)
        # Keep only peaks on usable frames.
        peaks = [p for p in peaks if usable[p]]
    else:
        peaks = []
    step_times = np.diff(np.array(peaks, dtype=float)) / fps if len(peaks) >= 2 else np.array([])
    n_steps = len(peaks)
    if len(step_times) >= 2:
        st_mean = float(np.mean(step_times))
        st_sd = float(np.std(step_times, ddof=1))
        st_cv = float(st_sd / st_mean) if st_mean else None
    else:
        st_mean = st_sd = st_cv = None

    # Gait speed is the median hip-centre speed over the walking bout, i.e.
    # between the first and last detected step. If fewer than 2 steps were
    # found, fall back to all usable frames and record that.
    if len(peaks) >= 2:
        lo, hi = peaks[0], peaks[-1]
        speed_ok = usable[lo:hi] & usable[lo + 1:hi + 1]
        speed_window = "walking bout (first to last detected step)"
        gait_speed_px = float(np.nanmedian(speed_px_s[lo:hi][speed_ok])) if speed_ok.any() else None
        gait_speed_bh = float(np.nanmedian(speed_bh_s[lo:hi][speed_ok])) if speed_ok.any() else None
    else:
        speed_ok = usable[:-1] & usable[1:]
        speed_window = "all usable frames (fewer than 2 steps detected)"
        gait_speed_px = float(np.nanmedian(speed_px_s[speed_ok])) if speed_ok.any() else None
        gait_speed_bh = float(np.nanmedian(speed_bh_s[speed_ok])) if speed_ok.any() else None

    return {
        "sequence": name,
        "n_frames": int(n),
        "fps_assumed": fps,
        "frames_pose_detected": int(detected.sum()),
        "frames_usable_ankle_vis_ge_0.5": int(usable.sum()),
        "usable_fraction": float(usable.mean()),
        "median_body_height_px": body_h_med,
        "n_steps_detected": int(n_steps),
        "segment_usable_ge_6_steps": bool(n_steps >= 6),
        "step_time_mean_s": st_mean,
        "step_time_sd_s": st_sd,
        "step_time_cv": st_cv,
        "gait_speed_px_per_s_median": gait_speed_px,
        "gait_speed_body_heights_per_s_median": gait_speed_bh,
        "gait_speed_window": speed_window,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="frame dirs or video files")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for path in args.inputs:
        name = os.path.basename(os.path.normpath(path))
        frames = load_frames(path)
        if not frames:
            print(f"SKIP {path}: no frames", file=sys.stderr)
            continue
        r = process_sequence(frames, args.fps, name)
        results.append(r)
        print(f"{name}: frames={r['n_frames']} usable={r['usable_fraction']:.3f} "
              f"steps={r['n_steps_detected']} usable_seg={r['segment_usable_ge_6_steps']} "
              f"gait_bh_s={r['gait_speed_body_heights_per_s_median']} "
              f"step_cv={r['step_time_cv']}")

    usable = [r for r in results if r["segment_usable_ge_6_steps"]]
    adl = [r for r in results if r["sequence"].startswith("adl-")]
    fall = [r for r in results if r["sequence"].startswith("fall-")]
    adl_speeds = [r["gait_speed_body_heights_per_s_median"] for r in adl
                  if r["gait_speed_body_heights_per_s_median"] is not None]
    adl_cvs = [r["step_time_cv"] for r in adl if r["step_time_cv"] is not None]
    agg = {
        "method": ("MediaPipe BlazePose (mp.solutions.pose, model_complexity=1); "
                   "visibility < 0.5 rejected; 5-frame moving-average smoothing; "
                   "steps = peaks of 2D inter-ankle distance (min separation 0.3 s, "
                   "prominence 0.05 body heights); usable segment needs >= 6 steps; "
                   "gait speed = median hip-centre displacement per second over the "
                   "walking bout (first to last detected step)"),
        "fps_assumed": args.fps,
        "fps_source": "URFD Kinect RGB recorded at 25 fps (Kwolek & Kepski 2014)",
        "n_sequences": len(results),
        "n_adl_sequences": len(adl),
        "n_fall_sequences": len(fall),
        "n_sequences_usable_ge_6_steps": len(usable),
        "sequences_usable_ge_6_steps": [r["sequence"] for r in usable],
        "mean_usable_fraction": float(np.mean([r["usable_fraction"] for r in results])) if results else None,
        "total_frames": int(sum(r["n_frames"] for r in results)),
        "total_frames_pose_detected": int(sum(r["frames_pose_detected"] for r in results)),
        "total_frames_usable": int(sum(r["frames_usable_ankle_vis_ge_0.5"] for r in results)),
        "steps_detected_per_sequence": {r["sequence"]: r["n_steps_detected"] for r in results},
        "gait_speed_body_heights_per_s_adl_only": {
            "note": ("ADL sequences only. Fall sequences are excluded: their hip-centre "
                     "speed is contaminated by the fall motion itself, not walking."),
            "n": len(adl_speeds),
            "values": [round(v, 3) for v in adl_speeds],
            "mean": float(np.mean(adl_speeds)) if adl_speeds else None,
            "sd": float(np.std(adl_speeds, ddof=1)) if len(adl_speeds) >= 2 else None,
            "min": float(np.min(adl_speeds)) if adl_speeds else None,
            "max": float(np.max(adl_speeds)) if adl_speeds else None,
        },
        "step_time_cv_adl_only": {
            "note": ("ADL sequences with at least 2 step intervals. Each CV rests on "
                     "only 2-4 intervals, so these values are unstable."),
            "n": len(adl_cvs),
            "values": [round(v, 3) for v in adl_cvs],
            "mean": float(np.mean(adl_cvs)) if adl_cvs else None,
            "min": float(np.min(adl_cvs)) if adl_cvs else None,
            "max": float(np.max(adl_cvs)) if adl_cvs else None,
        },
        "timed_walk_mae": None,
        "timed_walk_mae_note": ("No timed-walk ground truth exists in the UR Fall Detection "
                                 "Dataset, so no MAE vs timed walks can be computed. Gait speed "
                                 "is reported in body-heights/s and px/s only; no m/s figure "
                                 "is claimed because the camera is uncalibrated."),
        "per_sequence": results,
    }
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
