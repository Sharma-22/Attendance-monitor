"""Local webcam face-recognition attendance logger.

Press r to register a consenting attendee, q to quit. Attendance is recorded
once per person per local calendar day in attendance_data/attendance.csv.
"""
import csv
import json
import os
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "attendance_data")
SAMPLES_DIR = os.path.join(DATA_DIR, "face_samples")
IDENTITIES_PATH = os.path.join(DATA_DIR, "identities.json")
ATTENDANCE_PATH = os.path.join(DATA_DIR, "attendance.csv")
CAMERA_INDEX = 0
FACE_SIZE = (200, 200)
ENROLLMENT_SAMPLES = 25
MIN_ENROLLMENT_SAMPLES = 10
RECOGNITION_THRESHOLD = 70.0
CONFIRMATION_FRAMES = 5


def load_identities():
    if not os.path.exists(IDENTITIES_PATH):
        return {}
    with open(IDENTITIES_PATH, encoding="utf-8") as file:
        return {int(label): name for label, name in json.load(file).items()}


def save_identities(identities):
    with open(IDENTITIES_PATH, "w", encoding="utf-8") as file:
        json.dump({str(label): name for label, name in identities.items()},
                  file, ensure_ascii=False, indent=2)


def prepare_face(gray, box):
    x, y, width, height = box
    face = gray[y:y + height, x:x + width]
    face = cv2.equalizeHist(face)
    return cv2.resize(face, FACE_SIZE)


def load_recognizer(identities):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_images = []
    labels = []
    for label in identities:
        label_dir = os.path.join(SAMPLES_DIR, str(label))
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if not filename.lower().endswith(".png"):
                continue
            image = cv2.imread(os.path.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                face_images.append(image)
                labels.append(label)
    if face_images:
        recognizer.train(face_images, np.asarray(labels, dtype=np.int32))
        return recognizer
    return None


def ensure_attendance_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["date", "time", "name", "status"])


def load_today_attendance():
    today = datetime.now().date().isoformat()
    present = set()
    if os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("date") == today:
                    present.add(row["name"].casefold())
    return present


def record_attendance(name, present_today):
    key = name.casefold()
    if key in present_today:
        return False
    now = datetime.now()
    with open(ATTENDANCE_PATH, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([now.date().isoformat(), now.strftime("%H:%M:%S"),
                                   name, "Present"])
    present_today.add(key)
    print(f"Attendance marked: {name} at {now:%H:%M:%S}")
    return True


def show_attendance_popup(name, already_marked=False):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    message = f"{name} is present"
    if already_marked:
        message += "\nAttendance was already marked today."
    try:
        messagebox.showinfo("Attendance Marked", message, parent=root)
    finally:
        root.destroy()


def register_face(camera, detector, identities):
    name = input("Attendee name (blank to cancel): ").strip()
    if not name:
        print("Registration cancelled.")
        return
    print(f"Get {name}'s permission before saving face data.")
    if input("Type YES after they agree to face registration: ").strip().upper() != "YES":
        print("Registration cancelled; no face data saved.")
        return

    existing_label = next(
        (label for label, saved_name in identities.items()
         if saved_name.casefold() == name.casefold()),
        None,
    )
    label = existing_label or (max(identities, default=0) + 1)
    captured_faces = []
    last_capture = 0.0
    deadline = time.time() + 60
    print("Look at the camera. Hold still; press q in the camera window to cancel.")

    while len(captured_faces) < ENROLLMENT_SAMPLES and time.time() < deadline:
        ok, frame = camera.read()
        if not ok:
            print("Could not read from the camera during registration.")
            return
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(80, 80))
        for x, y, width, height in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)
        cv2.putText(frame, f"Registering {name}: {len(captured_faces)}/{ENROLLMENT_SAMPLES}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Face Attendance", frame)

        if len(faces) == 1 and time.time() - last_capture >= 0.25:
            captured_faces.append(prepare_face(gray, faces[0]))
            last_capture = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Registration cancelled.")
            return

    if len(captured_faces) < MIN_ENROLLMENT_SAMPLES:
        print(f"Registration failed: captured {len(captured_faces)} usable face samples; "
              f"need at least {MIN_ENROLLMENT_SAMPLES}.")
        return

    label_dir = os.path.join(SAMPLES_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)
    for index, face in enumerate(captured_faces):
        path = os.path.join(label_dir, f"face_{time.time_ns()}_{index}.png")
        if not cv2.imwrite(path, face):
            raise OSError(f"Could not save registered face sample: {path}")
    identities[label] = name
    save_identities(identities)
    print(f"Registered {name} with {len(captured_faces)} face samples.")


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    ensure_attendance_file()
    identities = load_identities()
    detector_path = os.path.join(cv2.data.haarcascades,
                                 "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(detector_path)
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {detector_path}")
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "Face recognition requires opencv-contrib-python. "
            "Install the project's requirements.txt."
        )

    recognizer = load_recognizer(identities)
    present_today = load_today_attendance()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened. Check CAMERA_INDEX or close other camera apps.")

    print(
        f"Attendance file: {ATTENDANCE_PATH}\n"
        f"Registered attendees: {len(identities)}\n"
        "Press r to register a consenting attendee; q to quit."
    )
    label_streaks = {}
    popup_shown_today = set()
    attendance_notice = ""
    attendance_notice_until = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(80, 80))
            recognized_faces = []
            seen_labels = set()
            for box in faces:
                match = None
                if recognizer is not None:
                    label, confidence = recognizer.predict(prepare_face(gray, box))
                    if confidence <= RECOGNITION_THRESHOLD and label in identities:
                        match = label
                        seen_labels.add(label)
                recognized_faces.append((box, match))

            for label in list(label_streaks):
                label_streaks[label] = label_streaks[label] + 1 if label in seen_labels else 0
            for label in seen_labels:
                label_streaks[label] = label_streaks.get(label, 0) + (
                    0 if label in label_streaks else 1
                )
            confirmed_labels = {
                label for label, streak in label_streaks.items()
                if streak >= CONFIRMATION_FRAMES
            }
            popup_name = None
            popup_already_marked = False

            for (x, y, width, height), label in recognized_faces:
                confirmed_name = identities[label] if label in confirmed_labels else None
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)
                if confirmed_name:
                    label_text = f"{confirmed_name} - PRESENT"
                    color = (0, 200, 0)
                elif recognizer is None:
                    label_text = "Face detected - register with r"
                    color = (0, 220, 220)
                elif label is not None:
                    label_text = f"Checking: {identities[label]}"
                    color = (0, 220, 220)
                else:
                    label_text = "Unknown face"
                    color = (0, 165, 255)
                cv2.putText(frame, label_text, (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if confirmed_name and confirmed_name.casefold() not in popup_shown_today:
                    was_recorded = record_attendance(confirmed_name, present_today)
                    popup_name = confirmed_name
                    popup_already_marked = not was_recorded
                    popup_shown_today.add(confirmed_name.casefold())
                    attendance_notice = (
                        f"{confirmed_name} is present"
                        if was_recorded else
                        f"{confirmed_name} is present - already marked today"
                    )
                    attendance_notice_until = time.time() + 5
            if time.time() < attendance_notice_until:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 55), (0, 100, 0), -1)
                cv2.putText(frame, attendance_notice, (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "r = register attendee   q = quit",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow("Face Attendance", frame)
            if popup_name:
                show_attendance_popup(popup_name, popup_already_marked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                register_face(camera, detector, identities)
                recognizer = load_recognizer(identities)
                label_streaks.clear()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
