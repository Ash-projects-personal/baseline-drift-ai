#!/usr/bin/env python3
"""Baseline Drift AI - additional experiments run 2026-11-11 (progress report).

Real runs on the same data and splits as scripts/upfall_experiments.py and
scripts/sisfall_experiments.py. Anchor checks: the logistic regression must
reproduce the already-reported numbers before any new number is trusted.

New experiments:
  A. UP-Fall model comparison on the same subject-wise split (test subjects
     15, 16, 17): logistic regression vs random forest vs gradient boosting
     vs RBF SVM.
  B. Leave-one-subject-out cross-validation (LOSO) on UP-Fall (logistic
     regression and random forest): mean accuracy across held-out subjects.
  C. SisFall model comparison on the same subject-wise split (test subjects
     SA22, SA23, SE14, SE15): logistic regression vs random forest vs
     gradient boosting.
  D. SisFall LOSO with logistic regression.
  E. Synthetic alert-rule sensitivity sweep: baseline window {14, 28} days x
     z threshold {1.5, 2.0, 2.5, 3.0} on the labeled synthetic dataset.
     The (28 days, z = 2) cell must reproduce 3/3 detected, 0 false alerts,
     lags 1 / 28 / 41 days.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(ROOT, "results")

# --- load UP-Fall feature machinery without running its main() --------------
src = open(os.path.join(BASE, "upfall_experiments.py")).read()
ns = {"__file__": os.path.join(BASE, "upfall_experiments.py")}
exec(src[: src.index("def main():")], ns)
parse_trial, trial_features = ns["parse_trial"], ns["trial_features"]
FALL_ACTIVITIES = ns["FALL_ACTIVITIES"]

results = {}

def logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced",
                                            max_iter=2000, random_state=42))

def metrics(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    return {"accuracy": round(float(accuracy_score(yte, pred)), 4),
            "f1": round(float(f1_score(yte, pred)), 4),
            "roc_auc": round(float(roc_auc_score(yte, proba)), 4)}

# --- UP-Fall feature table (identical assembly to upfall_experiments.py) ----
df = pd.read_parquet(os.path.join(ROOT, "datasets", "up-fall", "train.parquet"))
parts = df["sequence"].str.extract(r"subject_(\d+)_activity_(\d+)_trial_(\d+)").astype(int)
parts.columns = ["subject", "activity", "trial"]
df = pd.concat([df.reset_index(drop=True), parts], axis=1)
df["label"] = df["activity"].isin(FALL_ACTIVITIES).astype(int)
rows, n_excl = [], 0
for _, r in df.iterrows():
    arr = parse_trial(r["raw_data"])
    if not np.isfinite(arr).all():
        n_excl += 1
        continue
    f = trial_features(arr)
    f.update(subject=int(r["subject"]), label=int(r["label"]))
    rows.append(f)
feat = pd.DataFrame(rows)
fcols = [c for c in feat.columns if c not in ("subject", "label")]
X, y, subj = feat[fcols].values, feat["label"].values, feat["subject"].values
print(f"UP-Fall: used {len(feat)} trials, excluded {n_excl}, features {len(fcols)}")
assert len(feat) == 493 and n_excl == 66

test_m = np.isin(subj, [15, 16, 17])
assert int(test_m.sum()) == 99

# A. model comparison, same subject-wise split
comp = {}
comp["logistic_regression"] = metrics(logreg(), X[~test_m], y[~test_m], X[test_m], y[test_m])
print("UP-Fall LogReg anchor:", comp["logistic_regression"])
assert abs(comp["logistic_regression"]["accuracy"] - 0.9495) < 0.001, "UP-Fall anchor failed"
comp["random_forest"] = metrics(
    RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=2),
    X[~test_m], y[~test_m], X[test_m], y[test_m])
comp["gradient_boosting"] = metrics(
    GradientBoostingClassifier(random_state=42),
    X[~test_m], y[~test_m], X[test_m], y[test_m])
comp["svm_rbf"] = metrics(
    make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=42)),
    X[~test_m], y[~test_m], X[test_m], y[test_m])
results["upfall_model_comparison_subjectwise"] = {"n_test": 99, "models": comp}
print("UP-Fall comparison:", comp)

# B. UP-Fall LOSO
def loso(X, y, groups, make_clf):
    accs = {}
    for g in sorted(set(groups)):
        te = groups == g
        clf = make_clf()
        clf.fit(X[~te], y[~te])
        accs[str(g)] = round(float(accuracy_score(y[te], clf.predict(X[te]))), 4)
    vals = np.array(list(accs.values()))
    return {"per_subject_accuracy": accs,
            "mean_accuracy": round(float(vals.mean()), 4),
            "sd_accuracy": round(float(vals.std(ddof=1)), 4),
            "min_accuracy": round(float(vals.min()), 4)}

results["upfall_loso_logreg"] = loso(X, y, subj, logreg)
results["upfall_loso_random_forest"] = loso(
    X, y, subj, lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2))
print("UP-Fall LOSO LogReg:", results["upfall_loso_logreg"]["mean_accuracy"],
      "+/-", results["upfall_loso_logreg"]["sd_accuracy"])
print("UP-Fall LOSO RF:", results["upfall_loso_random_forest"]["mean_accuracy"],
      "+/-", results["upfall_loso_random_forest"]["sd_accuracy"])

# --- SisFall -----------------------------------------------------------------
sf = pd.read_parquet(os.path.join(OUT, "sisfall_features.parquet"))
scols = [c for c in sf.columns if c not in ("subject", "activity", "trial", "label", "n_samples")]
Xs, ys, ss = sf[scols].values, sf["label"].values, sf["subject"].values
test_s = np.isin(ss, ["SA22", "SA23", "SE14", "SE15"])
print(f"SisFall: {len(sf)} files, test n = {int(test_s.sum())}")
assert int(test_s.sum()) == 430
scomp = {}
scomp["logistic_regression"] = metrics(logreg(), Xs[~test_s], ys[~test_s], Xs[test_s], ys[test_s])
print("SisFall LogReg anchor:", scomp["logistic_regression"])
assert abs(scomp["logistic_regression"]["accuracy"] - 0.9884) < 0.001, "SisFall anchor failed"
scomp["random_forest"] = metrics(
    RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=2),
    Xs[~test_s], ys[~test_s], Xs[test_s], ys[test_s])
scomp["gradient_boosting"] = metrics(
    GradientBoostingClassifier(random_state=42),
    Xs[~test_s], ys[~test_s], Xs[test_s], ys[test_s])
results["sisfall_model_comparison_subjectwise"] = {"n_test": 430, "models": scomp}
print("SisFall comparison:", scomp)
results["sisfall_loso_logreg"] = loso(Xs, ys, ss, logreg)
print("SisFall LOSO LogReg:", results["sisfall_loso_logreg"]["mean_accuracy"],
      "+/-", results["sisfall_loso_logreg"]["sd_accuracy"])

# --- E. synthetic alert-rule sweep -------------------------------------------
sdf = pd.read_csv(os.path.join(ROOT, "synthetic", "SYNTHETIC_daily_features_180d.csv"),
                  comment="#")
truth = json.load(open(os.path.join(ROOT, "synthetic", "SYNTHETIC_ground_truth.json")))

def run_rule(window, thr):
    detected, lags, false_subj = 0, [], 0
    for s, g in sdf.groupby("subject"):
        g = g.sort_values("day")
        rec = g[g["recorded"] == 1]
        base = rec[rec["day"] <= window]
        gm, gs = base["gait_speed_ms"].mean(), base["gait_speed_ms"].std(ddof=1)
        cm, cs = base["step_time_cv_pct"].mean(), base["step_time_cv_pct"].std(ddof=1)
        cond = (((rec["gait_speed_ms"] - gm) / gs) < -thr) & \
               (((rec["step_time_cv_pct"] - cm) / cs) > thr)
        cond = cond.values
        days = rec["day"].values
        alerts = [int(days[i]) for i in range(1, len(rec)) if cond[i] and cond[i - 1]]
        inj = truth["subjects"][s]["injected_change"]
        if inj:
            if alerts:
                detected += 1
                lags.append(int(alerts[0] - inj["onset_day"]))
        elif alerts:
            false_subj += 1
    return {"detected_of_3": detected, "false_alert_subjects_of_7": false_subj,
            "lags_days": sorted(lags)}

sweep = {}
for w in (14, 28):
    for t in (1.5, 2.0, 2.5, 3.0):
        sweep[f"window_{w}_z_{t}"] = run_rule(w, t)
        print(f"sweep W={w} z={t}:", sweep[f"window_{w}_z_{t}"])
anchor = sweep["window_28_z_2.0"]
assert anchor["detected_of_3"] == 3 and anchor["false_alert_subjects_of_7"] == 0 \
    and anchor["lags_days"] == [1, 28, 41], f"sweep anchor failed: {anchor}"
results["synthetic_alert_rule_sweep"] = sweep

json.dump(results, open(os.path.join(OUT, "extra_experiments_results.json"), "w"), indent=2)
print("WROTE results/extra_experiments_results.json")
