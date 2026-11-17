# Evaluation plan

**Status: planned — no measured results are reported in this document.**

This plan evaluates the complete attendance prototype, including the enrollment references and matching rule. Upstream model results should not be presented as this application's accuracy.

## 1. Record the environment

Record the commit SHA, operating system, Python and package versions, CPU/GPU, webcam, actual frame resolution, enrolled-person count, and similarity threshold. Save package versions with `python -m pip freeze` after a successful run.

## 2. Keep enrollment and evaluation data separate

- Enroll only consenting participants and keep all face images private.
- Capture evaluation images in a different session from enrollment, including changes in lighting, distance, and pose.
- Include people who are not enrolled to measure unknown-person handling.
- Split development and final test data by capture session. Tune thresholds only on the development set; freeze the threshold before evaluating the final test set.
- Do not treat neighboring video frames as independent examples without explaining their correlation.

Record the number of people and samples in each split. Document collection conditions and any exclusions.

## 3. Measure recognition decisions

For each labeled test sample, record the true identity (or unknown), predicted identity, maximum similarity score, detection outcome, and test condition.

Report counts alongside rates:

| Metric | Definition | Result |
| --- | --- | --- |
| Detection coverage | Samples with a detected face / all face-containing samples | Not measured |
| Correct identification rate | Enrolled-person samples assigned the correct identity / all enrolled-person samples | Not measured |
| Wrong identity rate | Enrolled-person samples assigned a different enrolled identity / all enrolled-person samples | Not measured |
| Rejection rate for enrolled people | Enrolled-person samples returned as Unknown or not detected / all enrolled-person samples | Not measured |
| Unknown false acceptance rate | Unknown-person samples assigned an enrolled identity / all unknown-person samples | Not measured |

Also report the unknown false acceptance rate among **detected** unknown faces so missed detections do not hide matching errors. Break results down by lighting, pose, and distance where there are enough samples; state small-sample limitations.

## 4. Measure runtime

- Separate model download/load time from steady-state inference.
- Measure the entire per-frame recognition path after warm-up.
- Report median and 95th-percentile frame latency, observed FPS, frame count, face count, and enrolled-person count.
- Repeat measurements on the same hardware and report the range.
- The Tkinter callback delay is not a measured FPS result.

## 5. Check attendance behavior

Use temporary output files and synthetic names for these checks:

- First recognition creates a header and a single row.
- Repeated recognition and app restarts do not duplicate a person's row on the same date.
- Different people receive separate rows.
- A new date permits another row.
- A continuously running camera session crossing midnight also permits a new row. **Known gap:** the current session cache needs a fix before this can pass.
- Unknown people do not create attendance rows.
- Stop/Start and Exit release camera resources as expected.

## 6. Publish a reproducible result summary

Include the environment, split methodology, sample counts, threshold-selection process, measured results, failure cases, and exact commands used. Publish aggregated results without exposing participant photos, names, embeddings, or attendance records.

A short demo should show enrollment, a recognized person, an unknown person, and the resulting CSV with fictional identifiers. Label demonstrations separately from benchmark evidence.
