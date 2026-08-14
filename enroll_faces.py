"""
Smart Attendance System — Day 4: Enroll known faces (Person 2)
-------------------------------------------------------------------
Reads student photos from a dataset folder structured like:

    dataset/
        Aman/
            photo1.jpg
            photo2.jpg
        Priya/
            photo1.jpg
        ...

For each student, computes a face embedding for every photo, averages
them into one reference embedding, and saves everything to
embeddings.pkl for recognize_faces.py to use.

Run:
    pip install opencv-python numpy
    python enroll_faces.py

Tips for good dataset photos:
    - 2-4 photos per student, front-facing, decent lighting
    - One clear face per photo (extra faces in frame are skipped with a warning)
"""

import cv2
import os
import pickle
import numpy as np
from face_utils import load_detector, load_recognizer, get_embedding

DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "embeddings.pkl"


def process_student_folder(student_name, folder_path, detector, recognizer):
    embeddings = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  [skip] Couldn't read {filename}")
            continue

        h, w = image.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(f"  [skip] No face found in {filename}")
            continue

        if len(faces) > 1:
            print(f"  [warn] Multiple faces in {filename} — using the largest one")
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        embedding = get_embedding(image, faces[0], recognizer)
        embeddings.append(embedding)
        print(f"  [ok] Encoded {filename}")

    return embeddings


def main():
    if not os.path.isdir(DATASET_DIR):
        raise RuntimeError(
            f"Dataset folder '{DATASET_DIR}' not found. Create it with one "
            f"subfolder per student, each containing a few photos of them."
        )

    print("Loading models...")
    # Detector input size gets reset per-image in process_student_folder,
    # so the initial size here doesn't matter much.
    detector = load_detector(320, 320)
    recognizer = load_recognizer()

    known_embeddings = {}  # name -> averaged embedding vector

    student_folders = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    if not student_folders:
        raise RuntimeError(
            f"No student folders found inside '{DATASET_DIR}'. "
            f"Expected structure: dataset/StudentName/photo1.jpg"
        )

    for student_name in student_folders:
        folder_path = os.path.join(DATASET_DIR, student_name)
        print(f"\nProcessing {student_name}...")

        embeddings = process_student_folder(student_name, folder_path, detector, recognizer)

        if not embeddings:
            print(f"  [warn] No usable photos for {student_name} — skipped entirely")
            continue

        # Average multiple photos into one reference embedding per student —
        # this makes recognition more robust than relying on a single photo.
        averaged = np.mean(embeddings, axis=0)
        known_embeddings[student_name] = averaged

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known_embeddings, f)

    print(f"\nDone. Enrolled {len(known_embeddings)} student(s) -> {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
