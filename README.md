# Smart Attendance System

A Python computer-vision prototype that recognizes enrolled faces through a webcam and records attendance in a local CSV file.

**YuNet detection · SFace embeddings · Cosine-similarity matching · Tkinter interface**

> **Project status:** Educational prototype. This repository integrates pretrained models; it does not train a new face-recognition model. No project-specific accuracy or speed benchmark has been published yet.

## What it does

- Builds one reference embedding per person from local enrollment photos.
- Detects faces and compares them against the enrolled references.
- Labels unmatched faces as `Unknown`.
- Writes `Name,Date,Time` to `attendance.csv`, checking for duplicate entries on the same local calendar date.
- Offers a desktop interface with Start, Stop, and Exit controls, plus a terminal-launched OpenCV view.

## How it works

1. **Enrollment:** Read photos from one folder per person, detect a face with YuNet, align and crop it, and extract an SFace embedding. Average that person's embeddings and save the references to `embeddings.pkl`.
2. **Recognition:** Detect each live face, extract its embedding, and compare it with every enrolled reference.
3. **Decision:** Select the highest-scoring reference that meets the configured cosine threshold (`0.363` in `face_utils.py`); otherwise return `Unknown`.
4. **Attendance:** Check the CSV for the person's name and today's date before appending a record. The camera loops also cache names already processed during the session.

The similarity score is **not a probability or an accuracy percentage**. The threshold is a starting configuration and needs evaluation on the intended data.

## Technology

| Component | Purpose |
| --- | --- |
| Python | Application and attendance logic |
| OpenCV | Webcam capture, YuNet detection, and SFace inference |
| NumPy | Averaging enrollment embeddings |
| Pillow and Tkinter | Desktop video display and controls |
| CSV | Local attendance storage |

## Quick start

You need Python, a webcam, and a desktop environment. Internet access is needed for the initial model downloads. Run commands from the repository root so the relative data and model paths resolve correctly.

### 1. Clone and create an environment

```bash
git clone https://github.com/theamanlab/smart-attendance-system.git
cd smart-attendance-system
python -m venv .venv
```

Activate it on **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on **macOS/Linux**:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m tkinter
```

The second command should open a small window; close it before continuing. Tkinter is supplied with many Python installations but may require an OS package on Linux. This project needs OpenCV's desktop window support; use `opencv-python`, not a headless build.

Dependencies are currently unpinned. A tested version matrix and lock file remain future work.

### 3. Prepare enrollment photos

Create folders such as `dataset/Person_A/` and `dataset/Person_B/`. Put 2–4 clear JPG, JPEG, or PNG photos in each folder.

Use one consenting person per image, varied lighting, and a clear face. The folder name becomes the attendance name. **If a photo contains multiple faces, the current code selects the largest one**, so inspect enrollment images carefully.

### 4. Build reference embeddings

```bash
python enroll_faces.py
```

This downloads missing YuNet and SFace model files and creates `embeddings.pkl`. Check the terminal output for skipped images and confirm that at least one person was enrolled. Re-run enrollment whenever you change the dataset, then restart the recognition app.

### 5. Start recognition

**Desktop interface:**

```bash
python attendance_ui.py
```

Click **Start** to open the camera, **Stop** to pause, and **Exit** to close.

**OpenCV window:**

```bash
python recognize_faces.py
```

Press **q** to quit. After both model files are downloaded and enrollment is complete, recognition can run offline.

## Repository guide

| File | Responsibility |
| --- | --- |
| `face_utils.py` | Model downloads, detection, alignment, embeddings, and similarity comparison |
| `enroll_faces.py` | Enrollment from labeled photo folders |
| `recognize_faces.py` | Live OpenCV recognition loop |
| `attendance_ui.py` | Tkinter application |
| `attendance.py` | CSV creation and duplicate checking |
| `day1_2_webcam_setup.py` | Earlier webcam experiment |
| `day3_face_detection.py` | Earlier detection experiment |
| `requirements.txt` | Third-party Python dependencies |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Planned evaluation procedure and results template |

Generated files are excluded from version control: `dataset/`, `embeddings.pkl`, `attendance.csv`, and the two ONNX model files.

## Example output

Illustrative CSV, not a measured test result:

```csv
Name,Date,Time
Person_A,2026-11-17,09:00:00
Person_B,2026-11-17,09:01:30
```

Dates and times use the computer's local clock.

## Limitations

- **No measured performance yet:** Accuracy, false accepts, false rejects, latency, and FPS need controlled evaluation. See the [evaluation plan](docs/EVALUATION.md).
- **No liveness detection:** Photos or screens may fool the system; it is not suitable for identity verification or consequential attendance decisions without further safeguards and human review.
- **Single default webcam:** Camera index `0` and a requested resolution of 640 × 480 are hardcoded. Cameras returning another resolution may require the detector input size to be updated.
- **Session crossing midnight:** The in-memory cache tracks names rather than name/date pairs. Restart the application on a new day to allow a fresh attendance entry for an already-seen person.
- **Single-process storage:** CSV writes are not coordinated across concurrent application instances. Run one instance at a time.
- **Model download recovery:** Failed downloads can leave an incomplete file. If model loading fails, remove the affected ONNX file and retry with a working connection.
- **Enrollment quality:** Poor lighting, pose changes, occlusion, and incorrect enrollment images can cause errors. Empty enrollment results should be corrected before running recognition.

## Data handling

Use only enrollment images you have permission to process. Keep photos, embeddings, and attendance records local and out of screenshots or commits. Embeddings are biometric data, not anonymized records.

Only load an `embeddings.pkl` file generated by your own trusted enrollment run: Python pickle can execute code when loading an untrusted file. Removing a person requires removing their photo folder and rebuilding the embeddings; existing CSV records must be handled separately.

## Next improvements

- [ ] Publish measured recognition and latency results using held-out images.
- [ ] Add a short demo with consenting participants or appropriately anonymized visuals.
- [ ] Make camera selection and similarity thresholds configurable.
- [ ] Handle date rollover, actual camera dimensions, and interrupted downloads.
- [ ] Add automated tests for attendance and recognition decision logic.
- [ ] Record tested dependency versions and provide a lock file.

## Model credits

The project uses pretrained [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) models from OpenCV Zoo. Consult the upstream model directories for their documentation and license terms.
