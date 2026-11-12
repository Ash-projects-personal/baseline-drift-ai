#!/usr/bin/env python3
"""Build the CSCE 5280 Project Progress Report (Proposal #3).
Max 3 content pages + references. fpdf2. All numbers come from results/*.json.
"""
from fpdf import FPDF
import os

BASE = os.path.expanduser("~/workspace/your_files/project-progress-report")
OUT = os.path.join(BASE, "project-progress-report.pdf")

class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font("Helvetica", "B", 13)
            self.multi_cell(0, 5.5, "Baseline Drift AI: Detecting Fall Risk in Older Adults From Changes in Their Own Movement")
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 9.5)
            self.cell(0, 5, "Project Progress Report (Proposal #3)  |  CSCE / BMEN 5280, Fall 2026", new_x="LMARGIN", new_y="NEXT")
            self.cell(0, 5, "Ashish Rathnakar Shetty, sole student investigator  |  November 11, 2026", new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(2)
    def footer(self):
        self.set_y(-11)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Project Progress Report, Baseline Drift AI, Shetty, Nov 11, 2026  |  Page {self.page_no()}", align="C")

pdf = Report("P", "mm", "Letter")
pdf.set_auto_page_break(auto=True, margin=13)
pdf.set_margins(15, 13, 15)
pdf.add_page()

def h(text):
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)

def p(text):
    pdf.multi_cell(0, 4.7, text)
    pdf.ln(0.8)

def table(headers, rows, widths):
    pdf.set_font("Helvetica", "", 8.5)
    with pdf.table(col_widths=widths, text_align="LEFT", line_height=4.2) as t:
        hdr = t.row()
        for head in headers:
            hdr.cell(head)
        for row in rows:
            r = t.row()
            for cell in row:
                r.cell(cell)
    pdf.ln(1.5)
    pdf.set_font("Helvetica", "", 9.5)

# ---- A ----
h("A. Project Objective and Current Status")
p("Goal. Home monitoring that flags rising fall risk weeks before a fall happens, by watching for drift in a person's own movement instead of comparing them to a population average. The hypothesis is that drift from a person's own baseline predicts decline earlier and with fewer false alerts than a population-trained classifier. This semester establishes measurement quality and feasibility, not proof of fall prediction.")
p("Aim 1, build an individual movement baseline per participant (gait speed, stride variability, postural sway, activity, heart rate). On track. The camera pipeline from the Research Strategy is built exactly as proposed and has been run on real video (Section C). The housemate video and heart rate data is in hand, and I am still processing it and working on extracting meaningful features from it to make the model better. No household results are claimed yet; Section C uses public data.")
p("Aim 2, detect meaningful drift from each person's own baseline (daily deviation features; logistic regression as the starting model, compared against a population-trained classifier). On track. The classifier and the drift alert rule are built and tested. The personal-baseline advantage is not yet established: results are mixed across the two public datasets used for the supplementary analysis (Section C).")
p("Aim 3, pilot with older adults and evaluate usability (false-alert rates, adherence, clinician feedback). At risk. I am still processing the older-adult data and working on extracting meaningful features from it to make the model better. The pilot evaluation itself is not done, and consent and human-participant requirements still need instructor guidance. The Week 13 plan depends on that decision (Section E).")

# ---- B ----
h("B. Work Completed Versus the Original Plan")
p("The plan was: camera and watch pipeline in Weeks 1-2, baselines and the optional labeled classifier in Weeks 2-4, older-adult setup in Weeks 4-5. What is done: the camera pipeline (MediaPipe BlazePose, visibility below 0.5 rejected, 5-frame smoothing, gait speed and step-time variability, usable only with at least 6 visible steps) is built and run on 12 real RGB sequences from the UR Fall Detection Dataset; the optional labeled fall classifier is tested on two public IMU datasets; the drift alert rule is validated on labeled synthetic data. What is not done: the housemate and older-adult data is still being processed and is not yet in the results, there is no camera validation against timed walks yet, and the older-adult pilot evaluation has not run.")

# ---- Table 1 ----
pdf.set_font("Helvetica", "B", 9.5)
pdf.cell(0, 5.5, "Table 1. Planned Versus Actual Progress", new_x="LMARGIN", new_y="NEXT")
table(
    ["Workstream / Aim", "Planned by Week 10", "Actual Status and Evidence", "Next Action"],
    [
        ["Data / Sensing (Aim 1)", "Camera + watch pipeline Wks 1-2; baselines Wks 2-4", "Camera pipeline built and run on 12 URFD RGB sequences (1,902 frames). Household video + heart rate data in hand; processing and feature extraction in progress. Timed-walk MAE untested.", "Finish processing household data; timed walks for the MAE test (Wk 11)."],
        ["Model / System (Aim 2)", "Logistic regression baseline + drift rule", "Classifier tested as supplementary analysis: UP-Fall acc. 0.949, SisFall acc. 0.988 (subject-wise). Alert rule on labeled synthetic data: 3 of 3 changes detected, 0 false alerts.", "Apply drift rule to camera features; test personal z-scoring on longitudinal data."],
        ["Evaluation (Aims 1-2)", "Baselines + optional UP-Fall classifier Wks 2-4", "Camera usability measured: only 1 of 12 clips met the 6-step rule. ICC analysis on both IMU datasets; personal-vs-population comparison mixed (Section C).", "Longer recordings; finalize evaluation split on household data."],
    ],
    [30, 42, 62, 51],
)

# ---- C ----
h("C. Preliminary Results and Technical Evidence")
p("Main results: camera pipeline on real video. The pipeline was run on 12 RGB sequences from the UR Fall Detection Dataset (Kwolek and Kepski 2014): 6 daily-activity and 6 fall sequences, 1,902 frames total at 25 fps, 640x480, single adult actor, fixed camera. Activities were labeled by visual inspection of the frames, since no activity manifest is published with the files. All sequences are simulated activity, not real falls, and no older adults appear in them. Gait speed is reported in body-heights per second (hip-centre displacement divided by the subject's own pixel height) because the camera is uncalibrated; no metres-per-second figure is claimed.")
p("BlazePose detected a pose on 1,775 of 1,902 frames (93.3%). After the proposal's visibility rule (ankle and foot landmarks at or above 0.5), 1,292 frames stayed usable (67.9%; mean usable fraction 0.692 across sequences, range 0.48-1.00). Frames were lost mainly to feet hidden behind furniture and to the subject being small or partly out of frame at the room edges. Only 1 of the 12 sequences met the proposal's usability rule of at least 6 visible steps (fall-03, 7 steps). The clips are 4-9 seconds long with only 2-4 seconds of walking, so the clips contain only 1-7 detected steps (Fig. 1). This is a real finding about the method: short clips cannot satisfy the 6-step rule, so household recordings must be longer continuous walks.")
pdf.image(os.path.join(BASE, "figures/fig10_camera_pipeline.png"), x=(210 - 150) / 2, w=150)
pdf.set_font("Helvetica", "", 8.5)
pdf.multi_cell(0, 4.2, "Figure 1. Camera pipeline per URFD sequence (blue: daily activity, red: fall). Left: usable fraction with steps detected. Right: gait speed; fall bars are not walking speed because hip-centre motion during a fall is not gait. Only fall-03 met the 6-step rule.")
pdf.ln(1.5)
pdf.set_font("Helvetica", "", 9.5)
p("Gait speed on the 6 daily-activity sequences (median over each walking bout, first to last detected step) was 0.31 body-heights per second on average (SD 0.10, range 0.22-0.48). Step-time variability (coefficient of variation) on the 5 daily-activity sequences with at least 2 step intervals averaged 0.43 (range 0.04-0.80), but each value rests on only 2-4 intervals, so these numbers are unstable and are reported as a feasibility check, not as baselines. No timed-walk ground truth exists in this dataset, so the camera speed target (MAE at most 0.10 m/s vs timed walks) could not be tested; that test moves to household timed walks in Week 11.")
p("Supplementary analysis: IMU fall classifier (optional analysis named in the Research Strategy). On UP-Fall (Martinez-Villaseñor et al. 2019; 493 trials used from 15 subjects, 225 falls, 268 daily activities), a random forest on 77 whole-trial IMU features scored accuracy 0.990, F1 0.989, ROC-AUC 1.000 on a subject-wise split (test subjects 15-17; n = 99); the logistic regression starting model scored 0.949, and a random split leaks same-subject trials, so the subject-wise number is the defensible one. On SisFall (Sucerquia et al. 2017; 4,396 files, 38 subjects), the random forest scored accuracy 0.998, F1 0.997, ROC-AUC 1.000 (n = 430) against 0.988 for the logistic regression. Leave-one-subject-out cross-validation over every subject gave mean accuracy 0.996 (random forest) and 0.984 (logistic regression) on UP-Fall and 0.982 (logistic regression) on SisFall. People differ far more from each other than from themselves: mean ICC(1) for walking was 0.94 on UP-Fall (belt magnitude mean 0.99), and magnitude features stayed person-stable across daily activities (mean ICC 0.96 belt, 0.91 ankle, 0.88 wrist; variability features lower at 0.72, 0.61, 0.51). SisFall showed the same pattern. For the logistic model, scoring against a subject's own mean instead of the population mean helped slightly on UP-Fall (accuracy 0.949 to 0.960, false positives 5 to 2, at the cost of 2 missed falls) but hurt recall on SisFall (accuracy 0.988 to 0.963, recall 0.980 to 0.900, missed falls 3 to 15). The hypothesis is not established by these two datasets (Fig. 2).")
pdf.image(os.path.join(BASE, "figures/fig3_icc.png"), x=(210 - 126) / 2, w=126)
pdf.set_font("Helvetica", "", 8.5)
pdf.multi_cell(0, 4.2, "Figure 2. Within- vs between-subject variability, UP-Fall daily activities (3 trials per subject per activity, 15 subjects). ICC near 1 means subjects are stable and distinct. Magnitude features are highly person-stable; variability features are weaker.")
pdf.ln(1.5)
pdf.set_font("Helvetica", "", 9.5)
p("Drift-detector validation on labeled synthetic data. Three detectors were compared on a clearly labeled synthetic dataset, all using the same 28-day personal baseline: 10 subjects over 180 days, with gradual drift injected in two older subjects from day 120 and an abrupt change in one young subject at day 90. The proposed alert rule (gait slowing plus increased step variability beyond two baseline variation units on two consecutive recorded days) detected all 3 injected changes with 0 false alerts on 7 unchanged subjects, but took 28 and 41 days on the gradual drifts (1 day on the abrupt change). A one-sided CUSUM chart on the combined drift score also detected all 3 with 0 false alerts and cut the gradual lags to 11 and 12 days; an EWMA chart matched it, and CUSUM on gait alone raised a false alert, so the combined score is kept and CUSUM becomes the primary detector. This validates the pipeline mechanics only; it is not evidence about real participants.")
p("Failed and negative results. (1) A naive univariate deviation score was anti-predictive (AUC 0.06-0.07) and was discarded. (2) Two UP-Fall subjects lost their entire pocket sensor channel and 66 trials were excluded rather than imputed. (3) SisFall values are raw sensor counts, not calibrated units. (4) Only 1 of 12 video clips met the 6-step usability rule. (5) The camera MAE target could not be tested at all. Limitations: both IMU datasets are short lab recordings, not months of home data; neither contains older adults who fell; the video clips show one young adult actor in staged activity; the URFD copy used here has RGB only, no depth, gyroscope, or heart rate.")

pdf.set_font("Helvetica", "B", 9.5)
pdf.cell(0, 5.5, "Table 3. Key Measured Results at a Glance", new_x="LMARGIN", new_y="NEXT")
table(
    ["Experiment", "Data", "Headline Result"],
    [
        ["Camera pipeline (main)", "URFD, 12 sequences, 1,902 frames", "Pose 93.3%; usable 67.9%; gait 0.31 body-heights/s; 1 of 12 clips met the step rule"],
        ["IMU classifier", "UP-Fall, 493 trials, 15 subjects", "Random forest accuracy 0.990, F1 0.989, ROC-AUC 1.000; LOSO 0.996"],
        ["IMU classifier", "SisFall, 4,396 files, 38 subjects", "Random forest accuracy 0.998, F1 0.997, ROC-AUC 1.000"],
        ["Personal vs population", "UP-Fall and SisFall", "UP-Fall 0.949 to 0.960; SisFall 0.988 to 0.963; not established"],
        ["Drift detectors", "Synthetic, 10 subjects, 180 days", "CUSUM 3 of 3, 0 false alerts, lags 0, 11, 12 days; z rule lags 1, 28, 41"],
    ],
    [38, 47, 100],
)

# ---- D ----
h("D. Problems, Decisions, and Alternatives")
p("1. Public video clips are too short for the 6-step rule. Only 1 of 12 clips met it, because the walking portions last 2-4 seconds. Decision: keep the 6-step rule and collect longer household recordings (at least 30 seconds of continuous walking per session).")
p("2. The ankle-visibility rule keeps only 67.9% of frames. Feet are hidden behind furniture and the subject is small at the room edges. Decision: keep the 0.5 rule (it is the proposal's) and design household camera placement so the walking path keeps feet visible: fixed camera, eye-level, unobstructed path.")
p("3. No timed-walk ground truth and no metres per second. The public camera is uncalibrated and the dataset has no timed walks. Decision: the MAE test moves to household timed walks in Week 11, with gait speed reported in body-heights per second until a calibration exists. If timed walks cannot be arranged, the final report will state the MAE target as untested rather than substituting another number.")

# ---- E ----
h("E. Remaining Four-Week Work Plan")
pdf.set_font("Helvetica", "B", 9.5)
pdf.cell(0, 5.5, "Table 2. Updated Path to the Final Demonstration", new_x="LMARGIN", new_y="NEXT")
table(
    ["Week", "Specific Deliverable / Decision", "Responsible", "Evidence of Completion"],
    [
        ["11", "Process household video (30 s walks); timed walks; camera MAE", "Shetty", "Video files + MAE number; submitted Nov 11"],
        ["12", "Drift rule on camera features; cumulative-sum rule test", "Shetty", "Baseline figures + alert log + lag comparison"],
        ["13", "Older-adult pilot setup (consent + 7 baseline days) or reduced-scope decision", "Shetty + instructor guidance", "Consent records or written scope decision"],
        ["14 Final", "Demonstration + final report and presentation", "Shetty", "Demo + final PDF + slides"],
    ],
    [18, 72, 32, 63],
)
p("Priorities are achievable validation over new features: longer household walks first, then the MAE test, then the drift rule on real longitudinal data, then the pilot if consent clears; if it does not, the final demonstration uses the household baselines plus the public-video pipeline, and the report will say so.")

# ---- F ----
h("F. Individual Contributions")
p("Ashish Rathnakar Shetty, sole student investigator: all work in this report, including the camera pipeline, the IMU analyses, the synthetic validation, and this report. No other contributors. No participant data appears in the results in this report: the housemate and older-adult data is still being processed, and I am working on extracting meaningful features from it to make the model better; the older-adult pilot evaluation has not started. Remaining responsibilities are all mine (Weeks 11-14, Table 2).")

# ---- References (allowed to follow the 3 pages) ----
pdf.add_page()
h("References")
pdf.set_font("Helvetica", "", 9.5)
for ref in [
    "Kwolek, B., Kepski, M. Human fall detection on embedded platform using depth maps and wireless accelerometer. Computer Methods and Programs in Biomedicine, 2014, 117(3), 489-501. (UR Fall Detection Dataset.)",
    "Martinez-Villaseñor, L., Ponce, H., Brieva, J., Moya-Albor, E., Núñez-Martínez, J., Peñafort-Asturiano, C. UP-Fall Detection Dataset: A Multimodal Approach. Sensors, 2019, 19(9), 1988.",
    "Sucerquia, A., López, J. D., Vargas-Bonilla, J. F. SisFall: A Fall and Movement Dataset. Sensors, 2017, 17(1), 198.",
    "Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., Grundmann, M. BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204, 2020.",
    "Project code and analysis scripts: available as supplementary files on request (camera pipeline, UP-Fall and SisFall analyses, synthetic generator seed 20261110, drift pipeline). Datasets: UR Fall Detection Dataset (fenix.ur.edu.pl/~mkepski/ds/uf.html), UP-Fall (sites.google.com/up.edu.mx/har-up), SisFall (MDPI Sensors 17(1), 198).",
]:
    pdf.multi_cell(0, 4.7, ref)
    pdf.ln(1)

pdf.output(OUT)
print("wrote", OUT)
