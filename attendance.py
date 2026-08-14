"""
Smart Attendance System — Day 6: Attendance logging (Person 3)
-------------------------------------------------------------------
Handles writing recognized names + timestamps to a CSV file, and
makes sure the same student isn't marked twice on the same day.

Used by recognize_faces.py — you shouldn't need to run this file
directly.
"""

import csv
import os
from datetime import datetime

ATTENDANCE_FILE = "attendance.csv"


def _ensure_file_exists():
    """Create the CSV with a header row if it doesn't exist yet."""
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def _already_marked_today(name: str, today: str) -> bool:
    """Check the CSV for an existing entry for this name today."""
    if not os.path.exists(ATTENDANCE_FILE):
        return False

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == name and row["Date"] == today:
                return True

    return False


def mark_attendance(name: str) -> bool:
    """
    Marks attendance for `name` with the current date/time, unless
    they've already been marked today.

    Returns True if a new entry was written, False if they were
    already marked (so the caller can show a different message).
    """
    _ensure_file_exists()

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    if _already_marked_today(name, today):
        return False

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, time_now])

    return True
