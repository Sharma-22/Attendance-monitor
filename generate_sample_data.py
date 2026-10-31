"""
Creates a SAMPLE session CSV (simulated data) so you can test
analyze_posture.py without a camera. Real data comes from posture_monitor.py.
"""
import csv
import os
from datetime import datetime, timedelta

import numpy as np

np.random.seed(7)
os.makedirs("sessions", exist_ok=True)
start = datetime.now()
threshold, rows, t = 22.0, [], 0.0
bad_since, alert_on = None, False

while t < 20 * 60:                                   # 20-minute session
    minute = t / 60
    base = 12 + 14 * (minute / 20) + 6 * np.sin(minute / 2)   # slouches more over time
    angle = max(2, base + np.random.normal(0, 3))
    status = "Good" if angle <= threshold else "Bad"
    bad_since = (bad_since or t) if status == "Bad" else None
    alert = int(bad_since is not None and t - bad_since >= 5)
    rows.append([(start + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                 round(t, 2), round(angle, 2), threshold, status, alert, 0])
    t += 0.2

path = os.path.join("sessions", "posture_log_SAMPLE.csv")
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "timestamp", "elapsed_sec", "angle", "threshold", "status", "alert", "breaks_taken"
    ])
    w.writerows(rows)
print("Sample created:", path)
