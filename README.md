# Baseline Drift AI

Detecting fall risk in older adults from changes in their own movement.

Course project for CSCE/BMEN 5280 (AI for Wearables and Healthcare), University of North Texas. Sole investigator: Ashish Rathnakar Shetty.

Most fall systems try to detect a fall after it happens. This project tests a different idea: learn how a person normally moves, then watch for slow drift away from that personal baseline. A change from your own normal is the signal, not a score against a population average.

## What is built

1. Camera pipeline (the proposal's main method). BlazePose pose estimation on walking video, ankle visibility filtering, step detection from inter-ankle distance, and gait features per session: gait speed in body heights per second, step time, and step-time variability.
2. IMU fall classifiers (supplementary analysis). Whole-trial features from wearable accelerometers, with logistic regression as the starting model and a random forest as the stronger model. All evaluation is subject-wise: no person appears in both training and test data.
3. Personal baseline analysis. Within-person versus between-person variability (ICC), and a direct test of scoring each person against their own baseline instead of the population mean.
4. Drift detection. Daily gait features are compared against a personal baseline built from the first 28 days. Three detectors are implemented and compared: a two-consecutive-day z rule, a one-sided CUSUM chart, and an EWMA chart, both charts on a combined drift score (gait slowing plus rising step variability).

## Measured results

Camera pipeline on the UR Fall Detection Dataset (12 RGB sequences, 1,902 frames): pose detected on 93.3 percent of frames, 67.9 percent usable after the visibility rule, mean gait speed 0.31 body heights per second. Only 1 of 12 short clips met the 6-step usable-segment rule; at a 3-step minimum, 8 of 12 sequences and 5 of 6 walking sequences are usable. Household recordings therefore use at least 30 seconds of continuous walking per session.

IMU classifiers, subject-wise splits: random forest accuracy 0.990 on UP-Fall (493 trials, 15 subjects) and 0.998 on SisFall (4,396 files, 38 subjects); logistic regression 0.949 and 0.988. Leave-one-subject-out cross-validation: mean accuracy 0.996 (random forest) and 0.984 (logistic regression) on UP-Fall, 0.982 (logistic regression) on SisFall. Scoring against a personal baseline instead of the population mean helped slightly on UP-Fall (false positives 5 to 2) but hurt recall on SisFall, so the personal-baseline hypothesis is not established by these datasets.

Drift detection, validated on a clearly labeled synthetic dataset (10 simulated subjects, 180 days each, generated only to test the pipeline): the z rule detected all 3 injected changes with 0 false alerts, with lags of 1, 28, and 41 days. CUSUM detected all 3 with 0 false alerts and cut the gradual lags to 11 and 12 days. EWMA matched CUSUM. CUSUM on gait speed alone raised a false alert, which is why the combined score is used. These are pipeline validation results, not measurements of real participants. Household data collection is in progress and no participant data is in this repository.

A naive univariate deviation score was also tried and was anti-predictive (AUC 0.06 to 0.07). It was discarded and is kept in the results as a negative result.

## Data

Public datasets, not included in this repository: UP-Fall Detection Dataset (Martinez-Villasenor et al., Sensors 2019), SisFall (Sucerquia et al., Sensors 2017), and the UR Fall Detection Dataset (Kwolek and Kepski, 2014). The labeled synthetic validation set is included under synthetic/ with its ground truth file, so the detector comparison can be reproduced exactly.

## Layout

- scripts/camera_pipeline.py: BlazePose pipeline for the UR Fall sequences.
- scripts/upfall_experiments.py and scripts/sisfall_experiments.py: IMU features, classifiers, ICC, and personal-baseline tests.
- scripts/extra_experiments.py: model comparison and leave-one-subject-out cross-validation.
- scripts/synthetic_drift_pipeline.py: the z-rule drift pipeline on the synthetic validation set.
- scripts/drift_detector_comparison.py: z rule versus CUSUM versus EWMA, plus the step-threshold sensitivity analysis.
- scripts/process_household_video.py: runs the camera pipeline on any household video and appends one feature row per session to a CSV.
- scripts/generate_synthetic.py: generates the labeled synthetic validation set.
- results/: the JSON result files behind every number above.
- build_report.py: builds the project progress report PDF.

## Running

Install the packages in requirements.txt, place the public datasets under datasets/, then run the scripts in the order listed above. Every script writes its results to results/ and refuses to invent numbers: if an input file is missing, it stops.
