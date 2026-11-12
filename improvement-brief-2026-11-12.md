# Baseline Drift AI - Improvement Brief
Written 2026-11-12, after the progress report. Everything in section 1 was run for real on the project's own data and verified against the earlier result files before being written here.

## 1. Improvements already done and measured

### A better drift detector (the project's main weakness, now fixed)
The progress report's known weakness was detection lag: the two-consecutive-day z rule caught the gradual drifts only after 28 and 41 days. Two standard control-chart detectors were run on the same labeled synthetic data, with the same 28-day personal baseline and the same scoring (script: scripts/drift_detector_comparison.py; results: results/drift_detector_comparison.json):

- z rule (current): 3 of 3 changes detected, 0 false alerts, lags 1 / 28 / 41 days.
- CUSUM on the combined drift score: 3 of 3 detected, 0 false alerts, lags 0 / 11 / 12 days.
- EWMA on the combined drift score: 3 of 3 detected, 0 false alerts, lags 0 / 11 / 11 days.
- CUSUM on gait speed alone: 3 of 3 detected but 1 false-alert subject and 1 alert before onset, so the two-signal combined score stays in the design.

Design change: CUSUM becomes the primary drift detector for the final report and demo. The z rule stays as the simple baseline it is compared against. The gradual-drift lag drops from about a month to about 11 days at the same zero false alerts on this validation data. This is still synthetic validation; the claim stays limited to that until household data is processed.

### A stronger Aim 2 classifier
From the model comparison run (scripts/extra_experiments.py; results: results/extra_experiments_results.json):

- Random forest, subject-wise split: UP-Fall accuracy 0.990 (logistic regression 0.949), SisFall 0.998 (logistic regression 0.988).
- Leave-one-subject-out cross-validation: UP-Fall mean accuracy 0.996 for random forest and 0.984 for logistic regression; SisFall 0.982.

Design change: random forest becomes the Aim 2 classifier, with the logistic regression kept as the baseline and leave-one-subject-out as the headline evaluation.

### Step-rule sensitivity (turns a limitation into a result)
From the stored camera results: at the proposal's 6-step rule only 1 of 12 UR Fall sequences is usable and 0 of 6 walking sequences. At a 3-step minimum, 8 of 12 sequences and 5 of 6 walking sequences are usable; at 2 steps, 10 of 12 and all 6 walking sequences. The 6-step rule stays for baseline building (it protects baseline quality), and this sensitivity table goes into the final report so the choice is defended with numbers instead of asserted.

### Household data pipeline is built and verified
scripts/process_household_video.py runs the unchanged BlazePose pipeline on any video file or folder of frames and appends one feature row per recording (participant, date, usable fraction, steps, step-time variability, gait speed) to household/household_daily_features.csv. Verified: on the UR Fall adl-01 sequence it reproduced the stored pipeline output exactly (150 frames, 136 detected, 84 usable, 5 steps). The moment housemate recordings exist, processing is a single command per video, and the CUSUM detector runs on the resulting CSV the same way it ran on the validation data.

## 2. Recording protocol for the household data (from the camera findings)
- Fixed camera at about eye level, no handheld clips.
- Straight, unobstructed path; the participant walks across the frame, not toward the camera (feet leaving the lower frame was a main cause of lost frames).
- At least 30 seconds of continuous walking per session; the public clips failed the step rule because their walking portions last only 2 to 4 seconds.
- Same path and camera position every session, so day-to-day numbers stay comparable.

## 3. Final demonstration plan (Week 14)
- Live flow: play or record one walk, run the household processor, show the extracted features, then show that participant's baseline chart with the CUSUM statistic crossing its limit on the drift example.
- Figures for the final report: per-participant baseline chart with CUSUM overlay; detector comparison table (z rule vs CUSUM vs EWMA); classifier table with leave-one-subject-out results; step-rule sensitivity table.
- Fallback if the older-adult pilot does not clear consent and instructor guidance: demonstrate on household baselines plus the public-data pipeline, and say so plainly in the report, as the progress report already does.

## 4. What still needs Ash or other people
- Housemate recordings following the protocol above (the processing side is ready).
- Older-adult pilot: still gated on consent and the instructor's word on human-participant requirements. Nothing collected before that clears.
- A timed walk (known distance, stopwatch) during one household session would unlock the gait-speed error test (MAE) that the public data could not support.
