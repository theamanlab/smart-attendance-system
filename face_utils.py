"""
Smart Attendance System — shared face detection/recognition helpers
----------------------------------------------------------------------
Used by both enroll_faces.py (Day 4) and recognize_faces.py (Day 5).

Uses OpenCV 5's built-in DNN models:
  - FaceDetectorYN (YuNet)   -> finds faces + 5 landmarks
  - FaceRecognizerSF (SFace) -> turns an aligned face into a 128-d embedding

Both live in the MAIN opencv-python package — no dlib, no contrib,
no CMake/build-tools needed. Model files auto-download on first run.
"""

import cv2
import os
import urllib.request

DETECTOR_MODEL_PATH = "face_detection_yunet.onnx"
DETECTOR_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL_PATH = "face_recognition_sface.onnx"
RECOGNIZER_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# SFace's own recommended thresholds (from OpenCV Zoo docs)
COSINE_THRESHOLD = 0.363
L2_THRESHOLD = 1.128


def _ensure_downloaded(path: str, url: str):
    if os.path.exists(path):
        return
    print(f"Downloading model to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download model automatically ({e}). "
            f"Download manually from:\n  {url}\nand save as '{path}'."
        )


def load_detector(frame_width: int, frame_height: int) -> cv2.FaceDetectorYN:
    _ensure_downloaded(DETECTOR_MODEL_PATH, DETECTOR_MODEL_URL)
    return cv2.FaceDetectorYN.create(
        DETECTOR_MODEL_PATH, "", (frame_width, frame_height),
        score_threshold=0.7, nms_threshold=0.3, top_k=5000,
    )


def load_recognizer() -> cv2.FaceRecognizerSF:
    _ensure_downloaded(RECOGNIZER_MODEL_PATH, RECOGNIZER_MODEL_URL)
    return cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL_PATH, "")


def get_embedding(image, face_row, recognizer: cv2.FaceRecognizerSF):
    """
    face_row: one row from detector.detect() results (box + 5 landmarks + score).
    Aligns and crops the face, then returns its 128-d embedding vector.
    """
    aligned = recognizer.alignCrop(image, face_row)
    embedding = recognizer.feature(aligned)
    return embedding


def compare_embeddings(recognizer: cv2.FaceRecognizerSF, emb1, emb2):
    """
    Returns (is_match, cosine_score). Higher cosine score = more similar.
    Uses SFace's own recommended threshold for the match decision.
    """
    cosine_score = recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)
    is_match = cosine_score >= COSINE_THRESHOLD
    return is_match, cosine_score
