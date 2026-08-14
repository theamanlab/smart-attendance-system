"""
Smart Attendance System — Day 7: Simple UI (Person 4)
-------------------------------------------------------------------
A basic Tkinter window with Start / Stop / Exit buttons that wraps
everything from Days 1-6 into one app instead of a plain terminal
script.

Run:
    pip install pillow
    python attendance_ui.py

(You already have opencv-python, numpy, etc. from earlier days —
 pillow is the only new install needed, it's used to show the
 webcam feed inside the Tkinter window.)
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import pickle
import os

from face_utils import load_detector, load_recognizer, get_embedding, compare_embeddings
from attendance import mark_attendance

EMBEDDINGS_FILE = "embeddings.pkl"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")

        # --- state ---
        self.cap = None                # webcam, only open while running
        self.running = False           # is the camera loop active?
        self.detector = None
        self.recognizer = None
        self.known_faces = {}
        self.marked_this_session = set()

        # --- load known faces once at startup ---
        self.known_faces = self.load_known_faces()

        # --- build the window layout ---
        self.video_label = tk.Label(root, bg="black")
        self.video_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.status_var = tk.StringVar(value="Press Start to begin")
        tk.Label(root, textvariable=self.status_var, font=("Arial", 11)).grid(
            row=1, column=0, columnspan=3, pady=(0, 5)
        )

        self.start_btn = tk.Button(root, text="Start", width=12, command=self.start_camera)
        self.start_btn.grid(row=2, column=0, padx=5, pady=5)

        self.stop_btn = tk.Button(root, text="Stop", width=12, command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=2, column=1, padx=5, pady=5)

        self.exit_btn = tk.Button(root, text="Exit", width=12, command=self.on_exit)
        self.exit_btn.grid(row=2, column=2, padx=5, pady=5)

        tk.Label(root, text="Attendance marked this session:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=10
        )
        self.log_box = tk.Listbox(root, height=6, width=50)
        self.log_box.grid(row=4, column=0, columnspan=3, padx=10, pady=(0, 10))

        # Let the window's X button behave the same as Exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ---------- setup helpers ----------

    def load_known_faces(self):
        if not os.path.exists(EMBEDDINGS_FILE):
            messagebox.showerror(
                "Missing embeddings.pkl",
                f"'{EMBEDDINGS_FILE}' not found.\n\n"
                f"Run enroll_faces.py first to build it from your dataset folder, "
                f"then reopen this app.",
            )
            self.root.after(100, self.root.destroy)
            return {}
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)

    def identify_face(self, embedding):
        best_name, best_score = "Unknown", -1.0
        for name, known_embedding in self.known_faces.items():
            is_match, score = compare_embeddings(self.recognizer, embedding, known_embedding)
            if is_match and score > best_score:
                best_name, best_score = name, score
        return best_name, best_score

    # ---------- button actions ----------

    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam.")
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if self.detector is None:
            self.status_var.set("Loading models...")
            self.root.update()  # force the label to redraw before the (slow) first load
            self.detector = load_detector(FRAME_WIDTH, FRAME_HEIGHT)
            self.recognizer = load_recognizer()

        self.running = True
        self.status_var.set("Camera running...")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.update_frame()  # kicks off the repeating loop

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")
        self.status_var.set("Stopped. Press Start to resume.")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def on_exit(self):
        self.stop_camera()
        self.root.destroy()

    # ---------- the main camera loop ----------

    def update_frame(self):
        """Reads one frame, runs detection/recognition, updates the UI,
        then schedules itself to run again — this is what keeps the
        video feed 'live' without freezing the Tkinter window."""
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Failed to read from webcam.")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        _, faces = self.detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                embedding = get_embedding(frame, face, self.recognizer)
                name, score = self.identify_face(embedding)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if name != "Unknown" and name not in self.marked_this_session:
                    newly_marked = mark_attendance(name)
                    self.marked_this_session.add(name)
                    entry = f"{name} - marked just now" if newly_marked else f"{name} - already marked today"
                    self.log_box.insert(tk.END, entry)

        # Convert the OpenCV (BGR) frame to something Tkinter can display (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage-collected
        self.video_label.config(image=imgtk)

        # Schedule the next frame ~30ms from now (roughly 30fps)
        self.root.after(30, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
