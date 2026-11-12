#!/usr/bin/env python3
"""Figure for the rebuilt report: camera-pipeline results per URFD sequence.
Reads results/camera_pipeline_results.json; writes figures/fig10_camera_pipeline.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/workspace/your_files/project-progress-report")
with open(os.path.join(BASE, "results/camera_pipeline_results.json")) as f:
    d = json.load(f)

seqs = d["per_sequence"]
names = [r["sequence"].replace("-cam0-rgb", "") for r in seqs]
usable = [r["usable_fraction"] for r in seqs]
steps = [r["n_steps_detected"] for r in seqs]
gait = [r["gait_speed_body_heights_per_s_median"] for r in seqs]
is_adl = [n.startswith("adl-") for n in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

colors = ["#1f77b4" if a else "#d62728" for a in is_adl]
x = range(len(names))
ax1.bar(x, usable, color=colors)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Fraction of frames usable\n(ankle visibility >= 0.5)")
ax1.set_title("Usable frames per sequence (n = 1,902 frames)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
for i, s in enumerate(steps):
    ax1.text(i, usable[i] + 0.03, f"{s} steps", ha="center", fontsize=8)
ax1.axhline(0.692, color="gray", linestyle="--", linewidth=1)
ax1.text(0.1, 0.72, "mean 0.69", ha="left", fontsize=8, color="gray")

ax2.bar(x, gait, color=colors)
ax2.set_ylabel("Gait speed (body heights / s)\nmedian over walking bout")
ax2.set_title("Gait speed per sequence (fall seqs: not walking speed)")
ax2.set_xticks(list(x))
ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax2.set_ylim(0, 1.0)
for i, v in enumerate(gait):
    if v is not None:
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
# mark the only sequence meeting the >=6-step rule
ax2.annotate("only sequence with\n>= 6 steps (fall-03)", xy=(8, gait[8]),
             xytext=(4.2, 0.82), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="black"))

fig.tight_layout()
out = os.path.join(BASE, "figures/fig10_camera_pipeline.png")
fig.savefig(out, dpi=150)
print("wrote", out)
