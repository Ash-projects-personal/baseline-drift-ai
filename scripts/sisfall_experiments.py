#!/usr/bin/env python3
"""Baseline Drift AI - SisFall supplementary experiments (course progress report).

Data: SisFall, Sucerquia et al., Sensors 2017, 17(1), 198.
Local copy: ../datasets/sisfall/raw/content/gdrive/MyDrive/Datasets/SisFall_dataset/
38 subjects (23 young SA01-SA23, 15 elderly SE01-SE15), 19 ADL + 15 falls,
200 Hz, 9 columns = ADXL345 accel, ITG3200 gyro, MMA8451Q accel (raw counts).
Labels come from the file names (F* = fall, D* = ADL).

Mirrors the UP-Fall experiment design: logistic regression (Aim 2 starting
model) with a subject-wise split vs a random split; ICC(1) within- vs
between-subject variability on repeated ADL trials; population- vs
personal-reference z-score comparison.
All numbers printed here come from the files read by this script. Nothing is simulated.
"""
import json
import os
import re
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
DATA = os.path.join(ROOT, "datasets", "sisfall", "raw", "content", "gdrive",
                    "MyDrive", "Datasets", "SisFall_dataset")
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
CACHE = os.path.join(OUT, "sisfall_features.parquet")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

PAT = re.compile(r"^([DF]\d+)_([A-Z]+\d+)_R(\d+)\.txt$")
ACTIVITY_NAMES = {
    "D01": "walking slowly", "D02": "walking quickly", "D03": "jogging slowly",
    "D04": "jogging quickly", "D05": "stairs up/down slowly",
    "D06": "stairs up/down quickly", "D07": "sit half chair slowly",
    "D08": "sit half chair quickly", "D09": "sit low chair slowly",
    "D10": "sit low chair quickly", "D11": "sit, try up, collapse",
    "D12": "sit, lie slowly, sit again", "D13": "sit, lie quickly, sit again",
    "D14": "back to lateral, back", "D15": "stand, bend knees",
    "D16": "stand, bend no knees", "D17": "get into/out of car",
    "D18": "stumble while walking", "D19": "gentle jump",
    "F01": "fall fwd slip", "F02": "fall back slip", "F03": "fall lateral slip",
    "F04": "fall fwd trip", "F05": "fall fwd jog trip", "F06": "fall vertical faint",
    "F07": "fall faint, hands on table", "F08": "fall fwd getting up",
    "F09": "fall lateral getting up", "F10": "fall fwd sitting down",
    "F11": "fall back sitting down", "F12": "fall lateral sitting down",
    "F13": "fall fwd using hands", "F14": "fall back", "F15": "fall lateral",
}
# Column layout: 0-2 ADXL345 accel, 3-5 ITG3200 gyro, 6-8 MMA8451Q accel.
SENSORS = [("adxl", 0), ("gyro", 3), ("mma", 6)]


def parse_file(path: str):
    """Returns (array, truncated_flag). 17 of 4396 files carry one stray extra
    value on a final line (verified: e.g. F07_SA01_R01.txt has 3000 nine-field
    lines plus one single-field line); the incomplete final row is dropped."""
    with open(path, "r") as fh:
        # files use CRLF line endings and ';' line terminators; normalize all
        # separators to commas for np.fromstring
        txt = fh.read().replace(";", "").replace("\r", "").replace("\n", ",")
    vals = np.fromstring(txt, sep=",")
    truncated = False
    if vals.size % 9 != 0:
        vals = vals[: (vals.size // 9) * 9]
        truncated = True
    return vals.reshape(-1, 9), truncated


def file_features(arr: np.ndarray):
    feats = {}
    for c in range(9):
        x = arr[:, c]
        feats[f"ch{c}_mean"] = float(x.mean())
        feats[f"ch{c}_std"] = float(x.std())
        feats[f"ch{c}_min"] = float(x.min())
        feats[f"ch{c}_max"] = float(x.max())
    for name, start in SENSORS:
        mag = np.linalg.norm(arr[:, start:start + 3], axis=1)
        feats[f"{name}_mag_mean"] = float(mag.mean())
        feats[f"{name}_mag_std"] = float(mag.std())
        feats[f"{name}_mag_max"] = float(mag.max())
    return feats


def main():
    if os.path.exists(CACHE):
        feat = pd.read_parquet(CACHE)
        print(f"loaded cached features: {len(feat)} files")
        tf_path = os.path.join(OUT, "sisfall_truncated_files.json")
        truncated_files = json.load(open(tf_path)) if os.path.exists(tf_path) else []
    else:
        rows = []
        n_files = 0
        truncated_files = []
        for subj in sorted(os.listdir(DATA)):
            sdir = os.path.join(DATA, subj)
            if not os.path.isdir(sdir):
                continue
            for fn in sorted(os.listdir(sdir)):
                m = PAT.match(fn)
                if not m:
                    continue
                act, subject, rep = m.group(1), m.group(2), int(m.group(3))
                arr, truncated = parse_file(os.path.join(sdir, fn))
                if truncated:
                    truncated_files.append(fn)
                f = file_features(arr)
                f["subject"] = subject
                f["activity"] = act
                f["trial"] = rep
                f["label"] = 1 if act.startswith("F") else 0
                f["n_samples"] = int(arr.shape[0])
                rows.append(f)
                n_files += 1
        feat = pd.DataFrame(rows)
        feat.to_parquet(CACHE)
        json.dump(truncated_files, open(os.path.join(OUT, "sisfall_truncated_files.json"), "w"))
        print(f"parsed {n_files} files; {len(truncated_files)} files had one stray "
              f"extra value, final partial row dropped: {truncated_files}")

    feat_cols = [c for c in feat.columns
                 if c not in ("subject", "activity", "trial", "label", "n_samples")]
    X = feat[feat_cols].values
    y = feat["label"].values
    subjects = feat["subject"].unique()
    n_young = sum(1 for s in subjects if s.startswith("SA"))
    n_elder = sum(1 for s in subjects if s.startswith("SE"))
    print(f"files: {len(feat)}  subjects: {len(subjects)} ({n_young} young, {n_elder} elderly)")
    print(f"fall files: {int(y.sum())}  ADL files: {int((1 - y).sum())}")

    results = {
        "dataset": "SisFall (Sucerquia et al., Sensors 2017, 17(1), 198)",
        "source_dir": DATA,
        "n_files": int(len(feat)),
        "n_files_with_one_stray_extra_value_final_row_dropped": int(len(truncated_files)),
        "truncated_files": truncated_files,
        "n_subjects": int(len(subjects)),
        "n_young_subjects": int(n_young),
        "n_elderly_subjects": int(n_elder),
        "n_fall_files": int(y.sum()),
        "n_adl_files": int((1 - y).sum()),
        "n_features": len(feat_cols),
        "label_rule": "fall = files named F*, ADL = files named D*",
        "model": "StandardScaler + LogisticRegression(C=1.0, class_weight=balanced, max_iter=2000)",
        "features": "per-file whole-recording stats: per-channel mean/std/min/max (9 ch) + per-sensor magnitude mean/std/max (ADXL345 accel, ITG3200 gyro, MMA8451Q accel)",
        "units_note": "sensor values are raw integer counts from the files, not calibrated g or deg/s",
    }

    def evaluate(mask_train, mask_test, split_name, Xmat=None):
        Xm = X if Xmat is None else Xmat
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, class_weight="balanced",
                                             max_iter=2000, random_state=42))
        clf.fit(Xm[mask_train], y[mask_train])
        pred = clf.predict(Xm[mask_test])
        proba = clf.predict_proba(Xm[mask_test])[:, 1]
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
        dummy = DummyClassifier(strategy="most_frequent").fit(Xm[mask_train], y[mask_train])
        m["dummy_accuracy"] = float(accuracy_score(y[mask_test], dummy.predict(Xm[mask_test])))
        return m, clf, y[mask_test], proba

    # ---- 2a. subject-wise split: test SA22, SA23, SE14, SE15 ---------------
    test_subjects = ["SA22", "SA23", "SE14", "SE15"]
    m_subj = feat["subject"].isin(test_subjects).values
    res_subj, _, y_subj, p_subj = evaluate(
        ~m_subj, m_subj,
        "subject-wise (test SA22, SA23, SE14, SE15; train other 34 subjects)")
    # ---- 2a. random stratified split ----------------------------------------
    rng = np.random.RandomState(42)
    idx = np.arange(len(feat))
    test_idx = []
    for lab in (0, 1):
        pool = idx[y == lab]
        rng.shuffle(pool)
        test_idx.extend(pool[: int(round(0.2 * len(pool)))])
    m_rand = np.zeros(len(feat), dtype=bool)
    m_rand[np.array(test_idx)] = True
    res_rand, _, y_rand, p_rand = evaluate(~m_rand, m_rand,
                                           "random stratified 80/20 file split (seed 42)")
    results["experiment_2a"] = {"subject_wise": res_subj, "random_split": res_rand}
    print("\n== 2a subject-wise =="); print(json.dumps(res_subj, indent=2))
    print("\n== 2a random split =="); print(json.dumps(res_rand, indent=2))

    # ---- 2b. ICC on repeated ADL activities (5 trials per subject) ----------
    icc_features = ["adxl_mag_mean", "adxl_mag_std", "mma_mag_mean",
                    "mma_mag_std", "gyro_mag_mean", "gyro_mag_std"]
    icc_per_activity = {}
    for act in ["D05", "D07", "D08", "D11", "D12"]:
        sub = feat[feat["activity"] == act]
        scope_res = {}
        for f in icc_features:
            counts = sub.groupby("subject")[f].size()
            keep = counts[counts == 5].index
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
            k = 5
            icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else float("nan")
            scope_res[f] = {
                "n_subjects": int(n_subj),
                "between_subject_var_msb": float(msb),
                "within_subject_var_msw": float(msw),
                "variance_ratio_between_within": float(msb / msw) if msw > 0 else None,
                "icc1": float(icc),
            }
        icc_per_activity[ACTIVITY_NAMES[act] + f" ({act})"] = scope_res
    icc_mean = {f: float(np.mean([icc_per_activity[a][f]["icc1"]
                                  for a in icc_per_activity if f in icc_per_activity[a]]))
                for f in icc_features}
    results["experiment_2b_icc"] = {"per_activity": icc_per_activity,
                                    "mean_icc_across_activities": icc_mean}
    print("\n== 2b ICC =="); print(json.dumps(results["experiment_2b_icc"], indent=2))

    # ---- 2c. population vs personal reference, same model -------------------
    def z_transform(mode):
        Xz = np.zeros_like(X, dtype=float)
        train_adl = feat[(feat["label"] == 0) & (~feat["subject"].isin(test_subjects))]
        pop_mean = train_adl[feat_cols].mean()
        pop_std = train_adl[feat_cols].std(ddof=1).replace(0, np.nan)
        for i, (_, row) in enumerate(feat.iterrows()):
            if mode == "population":
                m, s = pop_mean, pop_std
            else:
                own = feat[(feat["subject"] == row["subject"]) & (feat["label"] == 0)]
                if row["label"] == 0:
                    own = own[own["trial"] != row["trial"]]
                m, s = own[feat_cols].mean(), own[feat_cols].std(ddof=1).replace(0, np.nan)
            Xz[i] = ((row[feat_cols] - m) / s).fillna(0).values
        return Xz

    arms = {}
    for mode in ("population", "personal"):
        Xz = z_transform(mode)
        res, _, _, _ = evaluate(~m_subj, m_subj,
                                f"subject-wise, {mode}-reference z-scores", Xmat=Xz)
        arms[mode + "_zscore"] = res
    arms["raw_features"] = res_subj
    results["experiment_2c_reference"] = arms
    print("\n== 2c reference comparison =="); print(json.dumps(arms, indent=2))

    with open(os.path.join(OUT, "sisfall_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote", os.path.join(OUT, "sisfall_results.json"))

    # ---- figures -------------------------------------------------------------
    plt.rcParams.update({"font.size": 11})
    # Fig 5: confusion matrices, both splits
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, res, title in [(axes[0], res_subj, "Subject-wise split\n(test SA22, SA23, SE14, SE15)"),
                           (axes[1], res_rand, "Random 80/20 file split")]:
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
    fig.suptitle("SisFall fall-vs-ADL logistic regression (whole-recording features, "
                 f"n={len(feat)} files)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig5_sisfall_confusion.png"), dpi=200)
    plt.close(fig)

    # Fig 6: ROC curves, both splits
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for y_true, proba, label in [(y_subj, p_subj, "subject-wise split"),
                                 (y_rand, p_rand, "random 80/20 split")]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate (ADL called fall)")
    ax.set_ylabel("True positive rate (falls detected)")
    ax.set_title("SisFall fall-vs-ADL logistic regression ROC")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_sisfall_roc.png"), dpi=200)
    plt.close(fig)

    # Fig 7: ICC bars
    acts = list(icc_per_activity.keys())
    x = np.arange(len(icc_features))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    width = 0.16
    for k, act in enumerate(acts):
        vals = [icc_per_activity[act][f]["icc1"] for f in icc_features]
        ax.bar(x + (k - (len(acts) - 1) / 2) * width, vals, width, label=act)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_mag_", "\n") for f in icc_features])
    ax.set_ylabel("ICC(1): between-subject share of variance")
    ax.set_ylim(0, 1.15)
    ax.set_title("Within- vs between-subject variability, SisFall ADL trials\n"
                 "(5 trials per subject; ICC near 1 = subjects stable and distinct)")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_sisfall_icc.png"), dpi=200)
    plt.close(fig)

    # Fig 8: reference comparison bars
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
    ax.set_title("SisFall fall-vs-ADL, subject-wise split (test SA22, SA23, SE14, SE15):\n"
                 "raw features vs population-reference vs personal-reference z-scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_sisfall_reference.png"), dpi=200)
    plt.close(fig)
    print("figures saved to", FIG)


if __name__ == "__main__":
    main()
