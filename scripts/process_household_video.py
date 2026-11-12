#!/usr/bin/env python3
"""Baseline Drift AI - household video processor.

Runs the project's BlazePose camera pipeline (scripts/camera_pipeline.py,
unchanged) on one household recording and appends one feature row to a CSV.
Use one row per participant per day; the drift detectors then run on the CSV
exactly as they run on the validation data.

Recording protocol (from the UR Fall sensitivity findings): fixed camera at
about eye level, unobstructed straight path, participant walking ACROSS the
frame, at least 30 seconds of continuous walking per session.

Usage:
  python3 scripts/process_household_video.py \
      --input path/to/video.mp4 --participant P01 --date 2026-11-15 \
      [--fps 30] [--out household/household_daily_features.csv]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
from camera_pipeline import load_frames, process_sequence  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDS = ["participant", "date", "source", "fps", "n_frames",
          "frames_pose_detected", "frames_usable", "usable_fraction",
          "n_steps_detected", "segment_usable_ge_6_steps",
          "step_time_mean_s", "step_time_cv",
          "gait_speed_body_heights_per_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file or folder of frames")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--fps", type=float, default=None,
                    help="frames per second; read from the video when not given")
    ap.add_argument("--out", default=os.path.join(ROOT, "household",
                                                  "household_daily_features.csv"))
    args = ap.parse_args()

    fps = args.fps
    if fps is None:
        if os.path.isfile(args.input):
            cap = cv2.VideoCapture(args.input)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        else:
            fps = 25.0

    frames = load_frames(args.input)
    if not frames:
        raise SystemExit(f"no frames read from {args.input}")
    name = f"{args.participant}_{args.date}"
    r = process_sequence(frames, fps, name)
    row = {
        "participant": args.participant,
        "date": args.date,
        "source": os.path.basename(args.input.rstrip("/")),
        "fps": fps,
        "n_frames": r["n_frames"],
        "frames_pose_detected": r["frames_pose_detected"],
        "frames_usable": r["frames_usable_ankle_vis_ge_0.5"],
        "usable_fraction": round(r["usable_fraction"], 4),
        "n_steps_detected": r["n_steps_detected"],
        "segment_usable_ge_6_steps": r["segment_usable_ge_6_steps"],
        "step_time_mean_s": r.get("step_time_mean_s"),
        "step_time_cv": r.get("step_time_cv"),
        "gait_speed_body_heights_per_s": r.get("gait_speed_body_heights_per_s_median"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    print("appended:", row)
    print("csv:", args.out)


if __name__ == "__main__":
    main()
