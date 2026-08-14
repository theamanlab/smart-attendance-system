# Smart Attendance System

A webcam-based attendance system that detects and recognizes faces in real time, then automatically logs attendance to a CSV file.

**Flow:** Webcam → Detect Face → Recognize Face → Mark Attendance (name + timestamp)

## Tech Stack

- Python
- OpenCV 5 (`opencv-python`) — face detection (YuNet) and face recognition (SFace)
- NumPy
- Pillow (for the Tkinter UI video display)
- Tkinter (built-in) — simple Start/Stop UI
- CSV — attendance storage

> **Note:** This project uses OpenCV's built-in DNN-based face detector (`FaceDetectorYN`) and recognizer (`FaceRecognizerSF`) instead of `face_recognition`/`dlib`. This avoids the CMake/Visual Studio build-tools dependency that `dlib` requires on Windows, while still giving accurate, modern face recognition.

## Project Structure

```
attendance_system/
├── face_utils.py           # Shared helper functions for detection/recognition
├── enroll_faces.py         # Builds embeddings.pkl from photos in dataset/
├── recognize_faces.py      # Terminal version — live recognition + attendance marking
├── attendance.py           # Handles writing to attendance.csv, prevents duplicate marks per day
├── attendance_ui.py        # Tkinter UI with Start / Stop / Exit buttons
├── day1_2_webcam_setup.py  # Early-stage webcam test script
├── day3_face_detection.py  # Early-stage face detection test script
├── .gitignore
└── README.md
```

The following are **not included in this repo** (see `.gitignore`) since they contain personal data or are auto-downloaded:
- `dataset/` — student face photos
- `embeddings.pkl` — face encodings built from those photos
- `attendance.csv` — attendance records
- `face_detection_yunet.onnx`, `face_recognition_sface.onnx` — model files (auto-download on first run)

## Setup

1. Clone the repo:
   ```
   git clone https://github.com/theamanlab/smart-attendance-system.git
   cd smart-attendance-system
   ```

2. Install dependencies:
   ```
   pip install opencv-python numpy pillow
   ```

3. Create a `dataset` folder with one subfolder per person, each containing 2–4 clear, front-facing photos:
   ```
   dataset/
     Aman/
       photo1.jpg
       photo2.jpg
     Priya/
       photo1.jpg
   ```

4. Build the face encodings (run once, or whenever you add a new person):
   ```
   python enroll_faces.py
   ```

## Usage

**Option A — Simple UI (recommended):**
```
python attendance_ui.py
```
Click **Start** to begin recognition, **Stop** to pause, **Exit** to close. Recognized names appear in the on-screen log and are saved to `attendance.csv`.

**Option B — Terminal version:**
```
python recognize_faces.py
```
Press `q` to quit.

Either way, the first run will auto-download the two required model files (~30MB total) — after that it works offline.

## How Attendance Marking Works

- Each recognized face is matched against the enrolled dataset using cosine similarity, with a threshold based on the SFace model's recommended value.
- A person is only marked **once per calendar date** — running the app multiple times in one day won't create duplicate rows, but a new day will always create a fresh entry.
- Output is a simple CSV: `Name, Date, Time`.

## Known Limitations

- Recognition accuracy depends on dataset photo quality (front-facing, decent lighting works best).
- Currently single-camera only (webcam index 0).
- No admin panel yet — enrolling new people requires re-running `enroll_faces.py`.
