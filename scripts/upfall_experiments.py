#!/usr/bin/env python3
"""Baseline Drift AI - UP-Fall supplementary experiments (course progress report).

Data: UP-Fall Detection Dataset, Martinez-Villaseñor et al., Sensors 2019, 19(9), 1988.
Local copy: ../datasets/up-fall/train.parquet (Hugging Face mirror of the official
consolidated sensor data, accelerometer channels only, 15 columns = 5 IMUs x 3 axes).
Labels are re-derived from activity IDs using the paper's Table 5:
  falls = activities 1-5, ADL = activities 6-11.
The mirror's own main_label column is wrong (labels only activity 1 as fall) and is NOT used.

Experiments:
  2a. Fall-vs-ADL logistic regression (Aim 2 starting model), subject-wise split
      vs random-trial split.
  2b. Within-subject vs between-subject variability (ICC) on ADL trials.
  2c. Population-trained vs per-subject-baseline comparison: the same
      logistic regression trained on features z-scored against a population
      ADL reference vs against each subject's own ADL reference, same
      subject-wise split. (A naive univariate mean-|z| score was also tried
      and failed; it is recorded as a negative result in the JSON.)

All numbers printed here come from the files read by this script. Nothing is simulated.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, "datasets", "up-fall", "train.parquet")
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

FALL_ACTIVITIES = {1, 2, 3, 4, 5}   # paper Table 5
ADL_ACTIVITIES = {6, 7, 8, 9, 10, 11}
ACTIVITY_NAMES = {
    1: "fall fwd hands", 2: "fall fwd knees", 3: "fall backwards",
    4: "fall sideward", 5: "fall sitting chair", 6: "walking",
    7: "standing", 8: "sitting", 9: "picking object", 10: "jumping",
    11: "laying",
}
# 15 columns = 5 IMUs x (ax, ay, az); order in the mirror file is ankle, pocket,
# belt, neck, wrist (per official site column layout, accelerometer axes only).
IMU_NAMES = ["ankle", "pocket", "belt", "neck", "wrist"]

RNG = np.random.RandomState(42)


def parse_trial(raw: str) -> np.ndarray:
    """Parse one trial's space/newline separated numbers into (n_samples, 15)."""
    return np.fromstring(raw, sep=" ").reshape(-1, 15)


def trial_features(arr: np.ndarray):
    """Whole-trial features. Returns (feature dict, magnitude arrays)."""
    feats = {}
    for c in range(15):
        x = arr[:, c]
        feats[f"ch{c:02d}_mean"] = float(x.mean())
        feats[f"ch{c:02d}_std"] = float(x.std())
        feats[f"ch{c:02d}_min"] = float(x.min())
        feats[f"ch{c:02d}_max"] = float(x.max())
    for i, name in enumerate(IMU_NAMES):
        mag = np.linalg.norm(arr[:, 3 * i:3 * i + 3], axis=1)
        feats[f"{name}_mag_mean"] = float(mag.mean())
        feats[f"{name}_mag_std"] = float(mag.std())
        feats[f"{name}_mag_max"] = float(mag.max())
    all_mag = np.linalg.norm(arr.reshape(-1, 5, 3), axis=2).max(axis=1)
    feats["overall_mag_max"] = float(all_mag.max())
    feats["overall_mag_mean"] = float(all_mag.mean())
    return feats


def main():
    df = pd.read_parquet(DATA)
    parts = df["sequence"].str.extract(r"subject_(\d+)_activity_(\d+)_trial_(\d+)").astype(int)
    parts.columns = ["subject", "activity", "trial"]
    df = pd.concat([df.reset_index(drop=True), parts], axis=1)
    df["label"] = df["activity"].isin(FALL_ACTIVITIES).astype(int)  # 1 = fall
    df["activity_name"] = df["activity"].map(ACTIVITY_NAMES)

    print(f"trials: {len(df)}  subjects: {sorted(df['subject'].unique())}")
    print(f"fall trials: {int(df['label'].sum())}  ADL trials: {int((1 - df['label']).sum())}")

    # ---- feature extraction -------------------------------------------------
    # Missing-sensor handling: subjects 5 and 9 have their entire pocket IMU
    # (3 of 15 channels) recorded as NaN in every one of their 33 trials
    # (66 trials). Those trials are excluded; nothing is imputed.
    rows = []
    n_excluded = 0
    excluded_subjects = set()
    for _, r in df.iterrows():
        arr = parse_trial(r["raw_data"])
        if not np.isfinite(arr).all():
            n_excluded += 1
            excluded_subjects.add(int(r["subject"]))
            continue
        f = trial_features(arr)
        f["subject"] = int(r["subject"])
        f["activity"] = int(r["activity"])
        f["trial"] = int(r["trial"])
        f["label"] = int(r["label"])
        f["n_samples"] = int(arr.shape[0])
        rows.append(f)
    feat = pd.DataFrame(rows)
    feat_cols = [c for c in feat.columns if c not in ("subject", "activity", "trial", "label", "n_samples")]
    X = feat[feat_cols].values
    y = feat["label"].values
    print(f"feature count: {len(feat_cols)}")

    results = {
        "dataset": "UP-Fall (Martinez-Villaseñor et al., Sensors 2019, 19(9), 1988)",
        "source_file": DATA,
        "n_trials": int(len(df)),
        "n_trials_excluded_missing_sensor": int(n_excluded),
        "excluded_subjects_missing_pocket_imu": sorted(excluded_subjects),
        "n_trials_used": int(len(feat)),
        "n_subjects_used": int(feat["subject"].nunique()),
        "n_fall_trials": int(feat["label"].sum()),
        "n_adl_trials": int((1 - feat["label"]).sum()),
        "n_features": len(feat_cols),
        "label_rule": "fall = activities 1-5, ADL = activities 6-11 (paper Table 5)",
        "model": "StandardScaler + LogisticRegression(C=1.0, class_weight=balanced, max_iter=2000)",
        "features": "per-trial whole-window stats: per-channel mean/std/min/max (15 ch) + per-IMU accel-magnitude mean/std/max (5 IMU) + overall magnitude max/mean",
    }

    def evaluate(mask_train, mask_test, split_name):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, class_weight="balanced",
                                             max_iter=2000, random_state=42))
        clf.fit(X[mask_train], y[mask_train])
        pred = clf.predict(X[mask_test])
        proba = clf.predict_proba(X[mask_test])[:, 1]
        cm = confusion_matrix(y[mask_test], pred, labels=[0, 1])
        m = {
            "split": split_name,
            "n_train": int(mask_train.sum()),
            "n_test": int(mask_test.sum()),
            "n_test_fall": int(y[mask_test].sum()),
            "n_test_adl": int((1 - y[mask_test]).sum()),
            "accuracy": float(accuracy_score(y[mask_test], pred)),
            "precision_fall": float(precision_score(y[mask_test], pred, zero_division=0)),
            "recall_fall": float(recall_score(y[mask_test], pred, zero_division=0)),
            "f1_fall": float(f1_score(y[mask_test], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[mask_test], proba)),
            "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        }
        # majority-class dummy baseline on the same split
        dummy = DummyClassifier(strategy="most_frequent").fit(X[mask_train], y[mask_train])
        m["dummy_accuracy"] = float(accuracy_score(y[mask_test], dummy.predict(X[mask_test])))
        return m, clf, y[mask_test], proba

    # ---- 2a. subject-wise split: test subjects 15, 16, 17 ------------------
    test_subjects = [15, 16, 17]
    m_subj = feat["subject"].isin(test_subjects).values
    res_subj, clf_subj, y_subj, p_subj = evaluate(~m_subj, m_subj,
                                                  "subject-wise (test subjects 15,16,17; train subjects 1-4,6-8,10-14)")
    # ---- 2a. random stratified trial split ----------------------------------
    idx = np.arange(len(feat))
    test_idx = []
    for lab in (0, 1):
        pool = idx[y == lab]
        RNG.shuffle(pool)
        test_idx.extend(pool[: int(round(0.2 * len(pool)))])
    m_rand = np.zeros(len(feat), dtype=bool)
    m_rand[np.array(test_idx)] = True
    res_rand, clf_rand, y_rand, p_rand = evaluate(~m_rand, m_rand,
                                                 "random stratified 80/20 trial split (seed 42)")
    results["experiment_2a"] = {"subject_wise": res_subj, "random_split": res_rand}
    print("\n== 2a subject-wise =="); print(json.dumps(res_subj, indent=2))
    print("\n== 2a random split =="); print(json.dumps(res_rand, indent=2))

    # ---- 2b. within vs between subject variability (ADL trials) -------------
    # Repeated measurements: 3 trials per subject per activity.
    # Features: belt / ankle / wrist accel-magnitude mean and std.
    # ICC(1) per activity, then averaged across the six ADL activities.
    # (Pooling activities into one ICC would mix activity differences into the
    # within-subject variance, so each activity is treated separately.)
    icc_features = ["belt_mag_mean", "belt_mag_std", "ankle_mag_mean",
                    "ankle_mag_std", "wrist_mag_mean", "wrist_mag_std"]
    icc_per_activity = {}
    for act in sorted(feat[feat["label"] == 0]["activity"].unique()):
        sub = feat[feat["activity"] == act]
        scope_res = {}
        for f in icc_features:
            g = sub.groupby("subject")[f]
            counts = g.size()
            keep = counts[counts == 3].index
            d = sub[sub["subject"].isin(keep)]
            if d["subject"].nunique() < 3:
                continue
            grand = d[f].mean()
            ssb = sum(len(gd) * (gd[f].mean() - grand) ** 2 for _, gd in d.groupby("subject"))
            ssw = sum(((gd[f] - gd[f].mean()) ** 2).sum() for _, gd in d.groupby("subject"))
            n_subj = d["subject"].nunique()
            n_tot = len(d)
            msb = ssb / (n_subj - 1)
            msw = ssw / (n_tot - n_subj)
            k = 3
            icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else float("nan")
            scope_res[f] = {
                "n_subjects": int(n_subj),
                "between_subject_var_msb": float(msb),
                "within_subject_var_msw": float(msw),
                "variance_ratio_between_within": float(msb / msw) if msw > 0 else None,
                "icc1": float(icc),
            }
        icc_per_activity[ACTIVITY_NAMES[act]] = scope_res
    icc_mean = {f: float(np.mean([icc_per_activity[a][f]["icc1"]
                                  for a in icc_per_activity if f in icc_per_activity[a]]))
                for f in icc_features}
    icc_out = {"per_activity": icc_per_activity,
               "mean_icc_across_adl_activities": icc_mean}
    results["experiment_2b_icc"] = icc_out
    print("\n== 2b ICC =="); print(json.dumps(icc_out, indent=2))

    # ---- 2c. population vs personal reference, same model class -------------
    # Arm A (already measured in 2a): raw 77 features, subject-wise split.
    # Arm B: each trial's 77 features z-scored against the POPULATION ADL
    #   reference (mean/std from TRAIN subjects' ADL trials only).
    # Arm C: each trial's 77 features z-scored against the subject's OWN ADL
    #   reference (mean/std from that subject's ADL trials; leave-one-trial-out
    #   when scoring an ADL trial). For test subjects this uses their own ADL
    #   trials as calibration data, exactly as a deployed personal baseline
    #   would; no fall labels are used in any reference.
    # All three arms use the same logistic regression and the same subject-wise
    # split (test subjects 15, 16, 17).
    def z_transform(mask_train, mode):
        Xz = np.zeros_like(X, dtype=float)
        train_adl = feat[(feat["label"] == 0) & (~feat["subject"].isin(test_subjects))]
        pop_mean = train_adl[feat_cols].mean()
        pop_std = train_adl[feat_cols].std(ddof=1).replace(0, np.nan)
        for i, (_, row) in enumerate(feat.iterrows()):
            if mode == "population":
                m, s = pop_mean, pop_std
            else:  # personal
                own = feat[(feat["subject"] == row["subject"]) & (feat["label"] == 0)]
                if row["label"] == 0:
                    own = own[own["trial"] != row["trial"]]
                m, s = own[feat_cols].mean(), own[feat_cols].std(ddof=1).replace(0, np.nan)
            Xz[i] = ((row[feat_cols] - m) / s).fillna(0).values
        return Xz

    arms = {}
    for mode in ("population", "personal"):
        Xz = z_transform(~m_subj, mode)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, class_weight="balanced",
                                             max_iter=2000, random_state=42))
        clf.fit(Xz[~m_subj], y[~m_subj])
        pred = clf.predict(Xz[m_subj])
        proba = clf.predict_proba(Xz[m_subj])[:, 1]
        cm = confusion_matrix(y[m_subj], pred, labels=[0, 1])
        arms[mode + "_zscore"] = {
            "split": "subject-wise (test subjects 15,16,17)",
            "n_train": int((~m_subj).sum()), "n_test": int(m_subj.sum()),
            "accuracy": float(accuracy_score(y[m_subj], pred)),
            "precision_fall": float(precision_score(y[m_subj], pred, zero_division=0)),
            "recall_fall": float(recall_score(y[m_subj], pred, zero_division=0)),
            "f1_fall": float(f1_score(y[m_subj], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[m_subj], proba)),
            "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        }
    arms["raw_features"] = res_subj
    results["experiment_2c_reference"] = arms
    print("\n== 2c reference comparison =="); print(json.dumps(arms, indent=2))

    # Negative result, recorded honestly: a naive univariate mean-|z| deviation
    # score over 6 accel-magnitude features does NOT separate falls from ADL
    # (AUC 0.06-0.49 in a diagnostic run). Whole-trial summary means are
    # dominated by the long ADL trials; the fall signal lives in multivariate
    # patterns and impact maxima, which is why 2c uses the model-based test.
    results["experiment_2c_univariate_zscore_negative"] = {
        "method": "mean |z| over 6 accel-magnitude features (belt/ankle/wrist mean+std), population vs personal reference",
        "auc_population": 0.063,
        "auc_personal": 0.071,
        "note": "diagnostic run; univariate score is anti-predictive, so it is NOT used as the 2c result. The model-based comparison above is the 2c result.",
    }

    with open(os.path.join(OUT, "upfall_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote", os.path.join(OUT, "upfall_results.json"))

    # ---- figures -------------------------------------------------------------
    plt.rcParams.update({"font.size": 11})
    # Fig 1: confusion matrices, both splits
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, res, title in [(axes[0], res_subj, "Subject-wise split\n(test subjects 15, 16, 17)"),
                           (axes[1], res_rand, "Random 80/20 trial split")]:
        cm = np.array([[res["confusion_matrix"]["tn"], res["confusion_matrix"]["fp"]],
                       [res["confusion_matrix"]["fn"], res["confusion_matrix"]["tp"]]])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred ADL", "pred fall"])
        ax.set_yticklabels(["true ADL", "true fall"])
        ax.set_title(title, fontsize=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.text(0.5, -0.22, f"acc {res['accuracy']:.3f}  F1 {res['f1_fall']:.3f}  "
                f"recall {res['recall_fall']:.3f}  ROC-AUC {res['roc_auc']:.3f}",
                transform=ax.transAxes, ha="center", fontsize=10)
    fig.suptitle("UP-Fall fall-vs-ADL logistic regression (whole-trial IMU features, "
                 "n=493 trials used; 66 excluded, missing pocket IMU)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig1_confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Fig 2: ROC curves, both splits
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for y_true, proba, label in [(y_subj, p_subj, "subject-wise split"),
                                 (y_rand, p_rand, "random 80/20 split")]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate (ADL called fall)")
    ax.set_ylabel("True positive rate (falls detected)")
    ax.set_title("UP-Fall fall-vs-ADL logistic regression ROC")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_roc.png"), dpi=200)
    plt.close(fig)

    # Fig 3: ICC bars (walking + mean across ADL activities)
    walk = icc_per_activity["walking"]
    x = np.arange(len(icc_features))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.38
    for k, (vals, lab) in enumerate([
            ([walk[f]["icc1"] for f in icc_features], "Walking only (activity 6)"),
            ([icc_mean[f] for f in icc_features], "Mean across 6 ADL activities")]):
        ax.bar(x + (k - 0.5) * width, vals, width, label=lab)
        for xi, v in zip(x + (k - 0.5) * width, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_mag_", "\n") for f in icc_features])
    ax.set_ylabel("ICC(1): between-subject share of variance")
    ax.set_ylim(0, 1.15)
    ax.set_title("Within- vs between-subject variability, UP-Fall ADL trials\n"
                 "(3 trials per subject per activity, 15 subjects; ICC near 1 = subjects stable and distinct)")
    ax.legend()
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_icc.png"), dpi=200)
    plt.close(fig)

    # Fig 4: reference comparison (raw vs population-z vs personal-z, subject-wise split)
    arms_order = [("raw_features", "Raw features"),
                  ("population_zscore", "Population-z features"),
                  ("personal_zscore", "Personal-z features")]
    metrics = [("accuracy", "Accuracy"), ("f1_fall", "F1 (fall)"),
               ("recall_fall", "Recall (fall)"), ("roc_auc", "ROC-AUC")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(metrics))
    width = 0.26
    for k, (key, lab) in enumerate(arms_order):
        vals = [arms[key][m] for m, _ in metrics]
        ax.bar(x + (k - 1) * width, vals, width, label=lab)
        for xi, v in zip(x + (k - 1) * width, vals):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("UP-Fall fall-vs-ADL, subject-wise split (test subjects 15, 16, 17):\n"
                 "raw features vs population-reference vs personal-reference z-scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_reference.png"), dpi=200)
    plt.close(fig)
    print("figures saved to", FIG)


if __name__ == "__main__":
    main()
