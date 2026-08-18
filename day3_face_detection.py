"""
Smart Attendance System — Day 3: Face Detection (Person 1)
-------------------------------------------------------------
Uses cv2.FaceDetectorYN (YuNet) instead of Haar Cascade.

Why: In OpenCV 5.0, CascadeClassifier was moved out of the main
opencv-python package into opencv-contrib-python's xobjdetect module,
and the contrib wheels currently ship without the haarcascade XML data
files. YuNet is a DNN-based detector that lives in the MAIN objdetect
module (no contrib needed), and is officially recommended by OpenCV
for new projects — it's also more accurate at angles and in low light,
which Haar Cascade struggles with.

Run:
    pip install opencv-python
    python day3_face_detection.py

First run downloads a small (~230KB) model file automatically.

Controls:
    q  -> quit
    s  -> save a snapshot (with rectangles drawn) to snapshots/
"""

import cv2
import os
import urllib.request
from datetime import datetime

SNAPSHOT_DIR = "snapshots"
MODEL_PATH = "face_detection_yunet.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that: (1) no other app is using the webcam, "
            "(2) camera permissions are granted, "
            "(3) try camera_index=1 if you have multiple cameras."
        )

    return cap


def ensure_model_downloaded():
    """Download the YuNet model file if it isn't already present."""
    if os.path.exists(MODEL_PATH):
        return

    print(f"Downloading face detection model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the model file automatically ({e}). "
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and place it in this folder as '{MODEL_PATH}'."
        )


def load_face_detector(frame_width: int, frame_height: int) -> cv2.FaceDetectorYN:
    ensure_model_downloaded()

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",                          # config path (not needed for onnx)
        (frame_width, frame_height), # input size — must match frame size
        score_threshold=0.7,         # confidence threshold (0-1)
        nms_threshold=0.3,           # overlap suppression
        top_k=5000,
    )
    return detector


def detect_faces(frame, detector: cv2.FaceDetectorYN):
    """
    Detect faces in a frame and draw rectangles around them.
    Returns (annotated_frame, list_of_face_boxes).
    Each face box is (x, y, w, h) — this is what Person 2 will crop
    and pass to the recognition step.
    """
    _, results = detector.detect(frame)
    boxes = []

    if results is not None:
        for face in results:
            x, y, w, h = face[:4].astype(int)
            confidence = face[-1]
            boxes.append((x, y, w, h))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Face {confidence:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

    return frame, boxes


def save_snapshot(frame) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    return path


def main():
    print("Opening webcam...")
    cap = open_webcam(camera_index=0)

    frame_width, frame_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print("Loading face detector...")
    detector = load_face_detector(frame_width, frame_height)

    print("Ready. Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        frame, faces = detect_faces(frame, detector)

        cv2.putText(
            frame, f"Faces: {len(faces)}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        cv2.imshow("Smart Attendance System - Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            path = save_snapshot(frame)
            print(f"Snapshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
