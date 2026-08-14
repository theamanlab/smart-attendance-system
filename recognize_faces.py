"""
Smart Attendance System — Day 5: Recognize faces live (Person 2)
-------------------------------------------------------------------
Loads embeddings.pkl (from enroll_faces.py), opens the webcam, detects
faces, and labels each one with the closest matching student name (or
"Unknown" if nothing matches closely enough).

Run:
    python recognize_faces.py

Requires embeddings.pkl to exist — run enroll_faces.py first.

Controls:
    q  -> quit
"""

import cv2
import pickle
import os
from face_utils import load_detector, load_recognizer, get_embedding, compare_embeddings
from attendance import mark_attendance

EMBEDDINGS_FILE = "embeddings.pkl"


def load_known_faces():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise RuntimeError(
            f"'{EMBEDDINGS_FILE}' not found. Run enroll_faces.py first "
            f"to build it from your dataset folder."
        )
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def identify_face(embedding, known_faces, recognizer):
    """
    Compares one live face embedding against every enrolled student.
    Returns (best_name, best_score). best_name is "Unknown" if no
    student clears the match threshold.
    """
    best_name = "Unknown"
    best_score = -1.0

    for name, known_embedding in known_faces.items():
        is_match, score = compare_embeddings(recognizer, embedding, known_embedding)
        if is_match and score > best_score:
            best_name = name
            best_score = score

    return best_name, best_score


def main():
    known_faces = load_known_faces()
    print(f"Loaded {len(known_faces)} known face(s): {', '.join(known_faces.keys())}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check it's not in use elsewhere.")

    frame_width, frame_height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    print("Loading models...")
    detector = load_detector(frame_width, frame_height)
    recognizer = load_recognizer()

    print("Ready. Press 'q' to quit.")

    # Track names already marked THIS SESSION so we don't check the CSV
    # on every single frame (mark_attendance also checks the CSV itself,
    # so this is just to avoid repeated file reads for the same person
    # sitting in front of the camera for several seconds).
    marked_this_session = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                embedding = get_embedding(frame, face, recognizer)
                name, score = identify_face(embedding, known_faces, recognizer)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

                if name != "Unknown" and name not in marked_this_session:
                    newly_marked = mark_attendance(name)
                    marked_this_session.add(name)
                    if newly_marked:
                        print(f"Attendance marked: {name}")
                    else:
                        print(f"{name} already marked today")

                if name != "Unknown" and name in marked_this_session:
                    cv2.putText(
                        frame, "Present", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

        cv2.imshow("Smart Attendance System - Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
