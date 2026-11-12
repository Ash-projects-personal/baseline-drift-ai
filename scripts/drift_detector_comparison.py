#!/usr/bin/env python3
"""Baseline Drift AI - drift detector comparison + step-threshold sensitivity.

Run 2026-11-12. Two real analyses:

1. Detector comparison on the labeled synthetic dataset (same 28-day
   per-subject baseline and the same scoring as synthetic_drift_pipeline.py):
   the current two-consecutive-day z rule vs one-sided CUSUM and EWMA
   control charts on the combined drift score d = (-z_gait + z_cv) / 2 and
   (for CUSUM) on gait alone. Standard SPC settings fixed in advance:
   CUSUM k = 0.5, h = 5 (tuned for a 1 SD shift); EWMA lambda = 0.2,
   3-sigma limits. The z rule must reproduce 3/3 detected, 0 false alerts,
   lags 1 / 28 / 41 before any comparison number is trusted.

2. Camera step-threshold sensitivity from results/camera_pipeline_results.json:
   how many of the 12 UR Fall sequences pass the usable-segment rule at
   step thresholds 2-8 (the report uses >= 6).
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, "results")
BASELINE_DAYS = 28

sdf = pd.read_csv(os.path.join(ROOT, "synthetic", "SYNTHETIC_daily_features_180d.csv"),
                  comment="#")
truth = json.load(open(os.path.join(ROOT, "synthetic", "SYNTHETIC_ground_truth.json")))

def subject_series():
    """Per subject: days, z_gait, z_cv on recorded days after baseline setup."""
    data = {}
    for s, g in sdf.groupby("subject"):
        g = g.sort_values("day")
        rec = g[g["recorded"] == 1]
        base = rec[rec["day"] <= BASELINE_DAYS]
        gm, gs = base["gait_speed_ms"].mean(), base["gait_speed_ms"].std(ddof=1)
        cm, cs = base["step_time_cv_pct"].mean(), base["step_time_cv_pct"].std(ddof=1)
        rec = rec.copy()
        rec["z_gait"] = (rec["gait_speed_ms"] - gm) / gs
        rec["z_cv"] = (rec["step_time_cv_pct"] - cm) / cs
        data[s] = rec
    return data

def score(alert_days_by_subject):
    detected, lags, false_subj, early = 0, [], 0, 0
    for s, alerts in alert_days_by_subject.items():
        inj = truth["subjects"][s]["injected_change"]
        if inj:
            onset = inj["onset_day"]
            after = [a for a in alerts if a >= onset]
            if any(a < onset for a in alerts):
                early += 1
            if after:
                detected += 1
                lags.append(int(after[0] - onset))
        elif alerts:
            false_subj += 1
    return {"detected_of_3": detected, "false_alert_subjects_of_7": false_subj,
            "lags_days": sorted(lags), "changed_subjects_with_pre_onset_alert": early}

data = subject_series()
results = {}

# --- current z rule (anchor) --------------------------------------------------
alerts = {}
for s, rec in data.items():
    cond = ((rec["z_gait"] < -2) & (rec["z_cv"] > 2)).values
    days = rec["day"].values
    alerts[s] = [int(days[i]) for i in range(1, len(rec)) if cond[i] and cond[i - 1]]
results["z_rule_two_consecutive"] = score(alerts)
print("z rule:", results["z_rule_two_consecutive"])
a = results["z_rule_two_consecutive"]
assert a["detected_of_3"] == 3 and a["false_alert_subjects_of_7"] == 0 \
    and a["lags_days"] == [1, 28, 41], "z-rule anchor failed"

# --- CUSUM on combined score and on gait alone --------------------------------
def cusum_alerts(values, days, k=0.5, h=5.0):
    S, out = 0.0, []
    for v, d in zip(values, days):
        S = max(0.0, S + v - k)
        if S > h:
            out.append(int(d))
            S = 0.0  # reset after alert so later alerts are also visible
    return out

alerts = {}
for s, rec in data.items():
    d = ((-rec["z_gait"] + rec["z_cv"]) / 2).values
    alerts[s] = cusum_alerts(d, rec["day"].values)
results["cusum_combined_k0.5_h5"] = score(alerts)
print("CUSUM combined:", results["cusum_combined_k0.5_h5"])

alerts = {}
for s, rec in data.items():
    alerts[s] = cusum_alerts((-rec["z_gait"]).values, rec["day"].values)
results["cusum_gait_only_k0.5_h5"] = score(alerts)
print("CUSUM gait only:", results["cusum_gait_only_k0.5_h5"])

# --- EWMA on combined score ----------------------------------------------------
def ewma_alerts(values, days, lam=0.2, L=3.0):
    sd_e = np.sqrt(lam / (2 - lam))
    E, out = 0.0, []
    for v, d in zip(values, days):
        E = lam * v + (1 - lam) * E
        if E > L * sd_e:
            out.append(int(d))
    return out

alerts = {}
for s, rec in data.items():
    d = ((-rec["z_gait"] + rec["z_cv"]) / 2).values
    alerts[s] = ewma_alerts(d, rec["day"].values)
results["ewma_combined_lambda0.2_L3"] = score(alerts)
print("EWMA combined:", results["ewma_combined_lambda0.2_L3"])

# --- step-threshold sensitivity -------------------------------------------------
cam = json.load(open(os.path.join(OUT, "camera_pipeline_results.json")))
steps = cam["steps_detected_per_sequence"]
# steps_detected_per_sequence maps sequence name -> steps; ADL names start 'adl'
sens = {}
for t in range(2, 9):
    ok_all = sum(1 for v in steps.values() if v >= t)
    ok_adl = sum(1 for name, v in steps.items() if name.startswith("adl") and v >= t)
    sens[f"threshold_{t}"] = {"usable_of_12": ok_all, "usable_adl_of_6": ok_adl}
results["step_threshold_sensitivity"] = sens
print("step sensitivity:", sens)
assert sens["threshold_6"]["usable_of_12"] == cam["n_sequences_usable_ge_6_steps"]

json.dump(results, open(os.path.join(OUT, "drift_detector_comparison.json"), "w"), indent=2)
print("WROTE results/drift_detector_comparison.json")
