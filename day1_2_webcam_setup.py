"""
Smart Attendance System — Day 1-2: Webcam Setup
-------------------------------------------------
Goal: open the webcam, display live video, and confirm everything works
before Person 1 adds face detection on top of this.

Run:
    pip install opencv-python
    python day1_2_webcam_setup.py

Controls:
    q  -> quit
    s  -> save a snapshot to snapshots/
"""

import cv2
import os
from datetime import datetime

SNAPSHOT_DIR = "snapshots"


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Try to open the webcam and fail loudly with a useful message if it can't."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that: (1) no other app is using the webcam, "
            "(2) camera permissions are granted, "
            "(3) try camera_index=1 if you have multiple cameras."
        )

    return cap


def save_snapshot(frame) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    return path


def main():
    print("Opening webcam...")
    cap = open_webcam(camera_index=0)

    # Optional: set resolution (helps keep processing fast later on)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened. Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Flip horizontally so it feels like a mirror (nicer UX for a demo)
        frame = cv2.flip(frame, 1)

        # This is where Person 1's face detection will hook in later,
        # e.g.: frame = detect_faces(frame)

        cv2.imshow("Smart Attendance System - Webcam Test", frame)

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
