# Face Recognition Attendance System

Offline webcam attendance demo. Recognized, pre-registered attendees are marked present once per day in a CSV file.

**Tech:** Python, OpenCV face detection, OpenCV LBPH face recognition

## Run attendance
```powershell
pip install -r posture_project\requirements.txt
python posture_project\posture_monitor.py
```
When running these commands from inside `posture_project`, use:
```powershell
pip install -r requirements.txt
python posture_monitor.py
```
`attendance_monitor.py` is the direct entry point for the same application.

1. Press `r` in the camera window to register a person.
2. Enter their name in the terminal and type `YES` only after they agree to face registration.
3. Have them face the camera while 25 face samples are captured.
4. Registered faces are identified from the camera, including multiple people in view; attendance is added to `attendance_data/attendance.csv` and a popup/window message confirms, for example, `Komal is present`. If already recorded that day, the popup says so.
5. Press `q` to quit. The same person is recorded at most once per calendar day.

The local `attendance_data/` folder contains face-crop samples and the names mapped to them. Restrict access to this folder and delete it when the data is no longer needed. This demo can be fooled by photos and is not suitable for secure access control or high-stakes attendance without additional safeguards and human review.

## Legacy posture tools
`analyze_posture.py` and `generate_sample_data.py` belong to the earlier posture-monitoring prototype; they are separate from attendance.

## Files
| File | Purpose |
|---|---|
| attendance_monitor.py | Face registration, recognition, and daily attendance logging |
| attendance_data/ | Local face samples, attendee names, and attendance CSV (created at runtime) |
| posture_monitor.py | Launches the attendance app using the original run command |
| requirements.txt | Python dependencies |
