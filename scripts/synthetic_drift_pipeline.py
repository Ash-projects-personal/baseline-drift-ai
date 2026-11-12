#!/usr/bin/env python3
"""Baseline/drift pipeline run on SYNTHETIC validation data.

Reads synthetic/SYNTHETIC_daily_features_180d.csv (simulated data, seed
20261110) and applies the project's proposed pipeline:
  - per-subject baseline from the first 28 days (recorded days only):
    mean and SD ("one baseline variation unit") of gait speed and of
    step-time variability (step_time_cv_pct).
  - daily deviation features: z = (value - baseline mean) / baseline SD.
  - alert rule (from the proposal): gait slowing AND increased step-time
    variability beyond two baseline variation units on two consecutive
    RECORDED days. "Consecutive recorded days" means consecutive in the
    recorded-day sequence; a missing day does not reset the streak.
    Alert day = the second day of the pair.
Detection is scored against the injected ground truth in
synthetic/SYNTHETIC_ground_truth.json (O02/O04 gradual drift from day 120,
Y03 abrupt change at day 90, all others no change).

Outputs: results/synthetic_drift_results.json and
figures/fig9_synthetic_gait_trajectories.png (labeled SYNTHETIC).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "synthetic", "SYNTHETIC_daily_features_180d.csv")
TRUTH = os.path.join(ROOT, "synthetic", "SYNTHETIC_ground_truth.json")
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

BASELINE_DAYS = 28
Z_GAIT = -2.0   # gait slowing beyond 2 baseline variation units
Z_CV = 2.0      # step-time variability increase beyond 2 baseline variation units


def main():
    df = pd.read_csv(DATA, comment="#")
    truth = json.load(open(TRUTH))
    assert "SYNTHETIC" in os.path.basename(DATA), "refusing to run on a non-synthetic file"
    print(f"rows: {len(df)}, subjects: {df['subject'].nunique()}")

    per_subject = {}
    all_alerts = []
    for subj, g in df.groupby("subject"):
        g = g.sort_values("day")
        rec = g[g["recorded"] == 1]
        base = rec[rec["day"] <= BASELINE_DAYS]
        assert len(base) >= 20, f"{subj}: only {len(base)} baseline days"
        b_gait_mean = float(base["gait_speed_ms"].mean())
        b_gait_sd = float(base["gait_speed_ms"].std(ddof=1))
        b_cv_mean = float(base["step_time_cv_pct"].mean())
        b_cv_sd = float(base["step_time_cv_pct"].std(ddof=1))
        rec = rec.copy()
        rec["z_gait"] = (rec["gait_speed_ms"] - b_gait_mean) / b_gait_sd
        rec["z_cv"] = (rec["step_time_cv_pct"] - b_cv_mean) / b_cv_sd
        rec["condition"] = (rec["z_gait"] < Z_GAIT) & (rec["z_cv"] > Z_CV)
        # alert on two consecutive recorded days meeting the condition
        cond = rec["condition"].values
        days = rec["day"].values
        alerts = []
        for i in range(1, len(rec)):
            if cond[i] and cond[i - 1]:
                alerts.append(int(days[i]))
        # detection lag vs injected onset
        inj = truth["subjects"][subj]["injected_change"]
        onset = inj["onset_day"] if inj else None
        first_alert = alerts[0] if alerts else None
        lag = (first_alert - onset) if (first_alert is not None and onset) else None
        # false-alert accounting: any alert in a subject with no injected change
        is_false = (inj is None and len(alerts) > 0)
        per_subject[subj] = {
            "group": truth["subjects"][subj]["group"],
            "n_recorded_days": int(len(rec)),
            "n_baseline_days": int(len(base)),
            "baseline_gait_speed_mean": round(b_gait_mean, 4),
            "baseline_gait_speed_sd": round(b_gait_sd, 4),
            "baseline_step_time_cv_mean": round(b_cv_mean, 3),
            "baseline_step_time_cv_sd": round(b_cv_sd, 3),
            "injected_change": inj,
            "onset_day": onset,
            "n_condition_days": int(cond.sum()),
            "alert_days": alerts,
            "n_alerts": len(alerts),
            "first_alert_day": first_alert,
            "detection_lag_days": lag,
            "false_alert_subject": bool(is_false),
        }
        all_alerts.extend([(subj, a) for a in alerts])
        print(f"{subj}: baseline gait {b_gait_mean:.3f}±{b_gait_sd:.3f} m/s, "
              f"CV {b_cv_mean:.2f}±{b_cv_sd:.2f}%, condition days {int(cond.sum())}, "
              f"alerts {alerts[:6]}{'...' if len(alerts) > 6 else ''}, "
              f"first alert {first_alert}, lag {lag}")

    n_drift = sum(1 for s in per_subject.values() if s["injected_change"])
    n_detected = sum(1 for s in per_subject.values()
                     if s["injected_change"] and s["first_alert_day"] is not None)
    false_subjects = [s for s, v in per_subject.items() if v["false_alert_subject"]]
    false_alert_days = sum(per_subject[s]["n_alerts"] for s in false_subjects)
    results = {
        "data_file": DATA,
        "seed": truth["seed"],
        "synthetic": True,
        "baseline_window_days": BASELINE_DAYS,
        "alert_rule": ("gait-speed z < -2 AND step-time-variability z > +2 "
                       "(z vs per-subject baseline mean/SD from first 28 recorded days) "
                       "on two consecutive recorded days; alert day = second day"),
        "per_subject": per_subject,
        "summary": {
            "n_subjects_with_injected_change": n_drift,
            "n_detected": n_detected,
            "n_missed": n_drift - n_detected,
            "false_alert_subjects": false_subjects,
            "n_false_alert_days": int(false_alert_days),
            "detection_lags": {s: v["detection_lag_days"] for s, v in per_subject.items()
                               if v["injected_change"]},
        },
    }
    with open(os.path.join(OUT, "synthetic_drift_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote", os.path.join(OUT, "synthetic_drift_results.json"))
    print("summary:", json.dumps(results["summary"], indent=2))

    # ---- figure: gait-speed trajectories, 10 panels --------------------------
    subjects = sorted(df["subject"].unique())
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=False)
    for ax, subj in zip(axes.ravel(), subjects):
        g = df[df["subject"] == subj].sort_values("day")
        rec = g[g["recorded"] == 1]
        ps = per_subject[subj]
        ax.plot(rec["day"], rec["gait_speed_ms"], ".", ms=3, color="#1f77b4",
                label="daily gait speed")
        ax.axhline(ps["baseline_gait_speed_mean"], color="green", lw=1.2,
                   label="baseline mean (first 28 d)")
        ax.axhline(ps["baseline_gait_speed_mean"] - 2 * ps["baseline_gait_speed_sd"],
                   color="orange", lw=1, ls="--", label="baseline mean - 2 SD")
        ax.axhline(ps["baseline_gait_speed_mean"] + 2 * ps["baseline_gait_speed_sd"],
                   color="orange", lw=1, ls="--")
        if ps["onset_day"]:
            ax.axvline(ps["onset_day"], color="red", lw=1.2, ls=":",
                       label=f"injected onset day {ps['onset_day']}")
        for a in ps["alert_days"]:
            ax.axvline(a, color="red", lw=1.5, alpha=0.7)
        if ps["alert_days"]:
            ax.plot([], [], color="red", lw=1.5, label="alert day")
        inj = ps["injected_change"]
        tag = f"{subj} ({ps['group']})"
        if inj:
            tag += f" - {inj['kind']} day {inj['onset_day']}"
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("day")
        ax.set_ylabel("gait speed (m/s)")
        ax.set_xlim(0, 182)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("SYNTHETIC DATA - daily gait speed, 10 simulated subjects (seed 20261110)\n"
                 "Alert rule: gait z < -2 AND step-time-variability z > +2 on 2 consecutive recorded days",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig9_synthetic_gait_trajectories.png"), dpi=200)
    plt.close(fig)
    print("figure saved to", os.path.join(FIG, "fig9_synthetic_gait_trajectories.png"))


if __name__ == "__main__":
    main()
