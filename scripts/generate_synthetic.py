#!/usr/bin/env python3
"""Generate SYNTHETIC daily-feature data for Baseline Drift AI pipeline validation.

SYNTHETIC DATA ONLY. Nothing in the output is a real measurement from any
person. The 10 subjects are invented; the numbers come from a seeded random
generator with hand-set group ranges. This file exists to validate the
baseline/drift pipeline (baseline estimation, deviation features, alert rule)
before any real participant data is collected. It must never be reported as
measured participant results.

Design:
  - 10 subjects: Y01-Y05 young roommates, O01-O05 older adults, 180 days each.
  - Daily features: resting_hr, hrv_sdnn_ms, walking_hr, gait_speed_ms,
    step_length_m, step_time_cv_pct, double_support_pct, step_count, sleep_hours.
    step_time_cv_pct is the daily step-time variability (coefficient of
    variation of step time, %); it is included because the project's alert
    rule is defined on gait slowing + increased step-time variability.
  - Group ranges (per-subject means drawn inside them):
      young gait speed 1.30-1.50 m/s, older 0.90-1.20 m/s
      young resting HR 55-70 bpm, older 60-75 bpm
      HRV (SDNN): young 40-70 ms, older 20-40 ms (lower in older adults)
      step length: young 0.70-0.80 m, older 0.55-0.65 m
      double support: young 20-25%, older 28-35%
      step-time CV: young 2.5-4.0%, older 4.0-6.5%
      step count: young 7000-11000/day, older 4000-7000/day
      sleep: 6.5-8.5 h both groups
    Older adults get larger day-to-day variability than young adults.
  - Coupling (keeps the data physically plausible): step length rises with
    gait speed; double support falls as gait speed rises; walking HR rises
    slightly on faster days.
  - Missing data: ~8% of days missing (young) / ~10% (older), plus 1-2
    watch-off gaps of 3-5 consecutive days per subject.
  - Weekend effect: step count x0.85 on Saturdays/Sundays.
  - Injected changes (the ground truth the pipeline must recover):
      O02, O04: GRADUAL drift from day 120: gait speed -0.003 m/s/day
        (total -0.18 by day 180), step-time CV +0.058 points/day (total +3.5),
        step length -0.001 m/day, double support +0.067 points/day,
        day-to-day SD x1.5 after onset.
      Y03: ABRUPT step change at day 90: gait speed -0.28 m/s, step-time CV
        +2.5 points, step length -0.08 m, double support +3.0 points,
        permanent (control case).
      All other subjects: no injected change.
Seed: 20261110 (numpy Generator). The seed and schedule are recorded in the
output README and results JSON.
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, "synthetic")
os.makedirs(OUT, exist_ok=True)

SEED = 20261110
N_DAYS = 180
rng = np.random.default_rng(SEED)

SUBJECTS = [f"Y0{i}" for i in range(1, 6)] + [f"O0{i}" for i in range(1, 6)]

# Injected changes: subject -> (kind, onset_day, params)
INJECTED = {
    "O02": ("gradual", 120, {"gait_per_day": -0.003, "cv_per_day": 0.058,
                             "step_len_per_day": -0.001, "ds_per_day": 0.067,
                             "sd_multiplier_after_onset": 1.5}),
    "O04": ("gradual", 120, {"gait_per_day": -0.003, "cv_per_day": 0.058,
                             "step_len_per_day": -0.001, "ds_per_day": 0.067,
                             "sd_multiplier_after_onset": 1.5}),
    "Y03": ("abrupt", 90, {"gait_shift": -0.28, "cv_shift": 2.5,
                           "step_len_shift": -0.08, "ds_shift": 3.0}),
}


def draw_subject_params(group: str):
    """Per-subject baseline means and day-to-day SDs, drawn inside the group ranges."""
    if group == "young":
        return {
            "gait_speed": (rng.uniform(1.30, 1.50), rng.uniform(0.03, 0.05)),
            "resting_hr": (rng.uniform(55, 70), rng.uniform(3, 5)),
            "hrv_sdnn": (rng.uniform(40, 70), rng.uniform(6, 9)),
            "walking_hr": (rng.uniform(95, 110), rng.uniform(4, 6)),
            "step_length": (rng.uniform(0.70, 0.80), rng.uniform(0.02, 0.03)),
            "step_time_cv": (rng.uniform(2.5, 4.0), rng.uniform(0.3, 0.5)),
            "double_support": (rng.uniform(20, 25), rng.uniform(1.0, 1.5)),
            "step_count": (rng.uniform(7000, 11000), rng.uniform(1200, 1800)),
            "sleep_hours": (rng.uniform(6.5, 8.5), rng.uniform(0.6, 0.9)),
            "miss_prob": 0.08,
        }
    return {
        "gait_speed": (rng.uniform(0.90, 1.20), rng.uniform(0.05, 0.08)),
        "resting_hr": (rng.uniform(60, 75), rng.uniform(4, 6)),
        "hrv_sdnn": (rng.uniform(20, 40), rng.uniform(4, 6)),
        "walking_hr": (rng.uniform(100, 120), rng.uniform(5, 7)),
        "step_length": (rng.uniform(0.55, 0.65), rng.uniform(0.02, 0.035)),
        "step_time_cv": (rng.uniform(4.0, 6.5), rng.uniform(0.4, 0.7)),
        "double_support": (rng.uniform(28, 35), rng.uniform(1.2, 1.8)),
        "step_count": (rng.uniform(4000, 7000), rng.uniform(900, 1400)),
        "sleep_hours": (rng.uniform(6.5, 8.5), rng.uniform(0.7, 1.0)),
        "miss_prob": 0.10,
    }


def main():
    rows = []
    truth = {}
    for subj in SUBJECTS:
        group = "young" if subj.startswith("Y") else "older"
        p = draw_subject_params(group)
        # watch-off gaps: 1-2 blocks of 3-5 consecutive missing days
        missing = np.zeros(N_DAYS, dtype=bool)
        missing |= rng.random(N_DAYS) < p["miss_prob"]
        n_gaps = int(rng.integers(1, 3))
        for _ in range(n_gaps):
            start = int(rng.integers(30, N_DAYS - 6))
            length = int(rng.integers(3, 6))
            missing[start:start + length] = True
        # keep the first 28 days mostly present so the baseline is estimable
        missing[:28] &= rng.random(28) < 0.02

        inj = INJECTED.get(subj)
        for day in range(1, N_DAYS + 1):
            if missing[day - 1]:
                rows.append({"subject": subj, "group": group, "day": day,
                             "recorded": 0, "resting_hr": np.nan, "hrv_sdnn_ms": np.nan,
                             "walking_hr": np.nan, "gait_speed_ms": np.nan,
                             "step_length_m": np.nan, "step_time_cv_pct": np.nan,
                             "double_support_pct": np.nan, "step_count": np.nan,
                             "sleep_hours": np.nan})
                continue
            d = day - 1  # 0-based
            # drift offsets for this day
            gait_off = cv_off = sl_off = ds_off = 0.0
            sd_mult = 1.0
            if inj is not None:
                kind, onset, par = inj
                if kind == "gradual" and day >= onset:
                    t = day - onset + 1
                    gait_off = par["gait_per_day"] * t
                    cv_off = par["cv_per_day"] * t
                    sl_off = par["step_len_per_day"] * t
                    ds_off = par["ds_per_day"] * t
                    sd_mult = par["sd_multiplier_after_onset"]
                elif kind == "abrupt" and day >= onset:
                    gait_off = par["gait_shift"]
                    cv_off = par["cv_shift"]
                    sl_off = par["step_len_shift"]
                    ds_off = par["ds_shift"]
            # gait speed first (drives the coupled features)
            gait = rng.normal(p["gait_speed"][0] + gait_off,
                              p["gait_speed"][1] * sd_mult)
            gait = float(np.clip(gait, 0.4, 2.0))
            # step length tied to gait speed (0.5 m per 1 m/s) plus own noise
            sl = (p["step_length"][0] + sl_off
                  + 0.5 * (gait - p["gait_speed"][0])
                  + rng.normal(0, p["step_length"][1]))
            sl = float(np.clip(sl, 0.3, 1.0))
            # double support falls as gait rises (8 points per 1 m/s)
            ds = (p["double_support"][0] + ds_off
                  - 8.0 * (gait - p["gait_speed"][0])
                  + rng.normal(0, p["double_support"][1] * sd_mult))
            ds = float(np.clip(ds, 10, 50))
            cv = float(np.clip(rng.normal(p["step_time_cv"][0] + cv_off,
                                          p["step_time_cv"][1] * sd_mult), 0.5, 20))
            # weekend step-count dip (day 1 = a Monday)
            weekend = 0.85 if (day % 7 in (6, 0)) else 1.0
            steps = float(np.clip(rng.normal(p["step_count"][0] * weekend,
                                             p["step_count"][1]), 500, 25000))
            rows.append({
                "subject": subj, "group": group, "day": day, "recorded": 1,
                "resting_hr": float(np.clip(rng.normal(*p["resting_hr"]), 40, 110)),
                "hrv_sdnn_ms": float(np.clip(rng.normal(*p["hrv_sdnn"]), 5, 150)),
                "walking_hr": float(np.clip(
                    rng.normal(p["walking_hr"][0] + 8.0 * (gait - p["gait_speed"][0]),
                               p["walking_hr"][1]), 60, 180)),
                "gait_speed_ms": gait,
                "step_length_m": sl,
                "step_time_cv_pct": cv,
                "double_support_pct": ds,
                "step_count": steps,
                "sleep_hours": float(np.clip(rng.normal(*p["sleep_hours"]), 3, 12)),
            })
        truth[subj] = {
            "group": group,
            "baseline_params": {k: {"mean": float(v[0]), "day_to_day_sd": float(v[1])}
                                for k, v in p.items() if k != "miss_prob"},
            "miss_prob": p["miss_prob"],
            "injected_change": ({"kind": inj[0], "onset_day": inj[1],
                                 "params": inj[2]} if inj else None),
            "n_recorded_days": int((~missing).sum()),
            "n_missing_days": int(missing.sum()),
        }

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT, "SYNTHETIC_daily_features_180d.csv")
    header = (
        "# SYNTHETIC DATA - simulated for pipeline validation only, NOT real participant data.\n"
        "# Generated by scripts/generate_synthetic.py, seed 20261110. No real person was measured.\n"
        "# Injected ground truth: O02 and O04 gradual drift from day 120; Y03 abrupt change at day 90.\n"
        "# Never report these numbers as measured participant results.\n"
    )
    with open(out_csv, "w") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    with open(os.path.join(OUT, "SYNTHETIC_ground_truth.json"), "w") as fh:
        json.dump({"seed": SEED, "n_days": N_DAYS, "subjects": truth}, fh, indent=2)
    print(f"wrote {out_csv} ({len(df)} rows, {df['recorded'].sum()} recorded days)")
    for s, t in truth.items():
        print(f"  {s} ({t['group']}): recorded {t['n_recorded_days']}/180, "
              f"injected={t['injected_change']['kind'] + ' day ' + str(t['injected_change']['onset_day']) if t['injected_change'] else 'none'}")


if __name__ == "__main__":
    main()
