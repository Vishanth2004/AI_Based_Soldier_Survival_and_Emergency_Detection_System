"""
AI-Based Soldier Survival & Emergency Detection System
Data Collection Server — Receives sensor data from ESP32 over WiFi

Features collected:
  ECG:  bpm, hrv_sdnn, hrv_rmssd
  IMU:  smv, dynamic_accel, impact, pitch, roll, gx, gy, gz, movement_var
  GPS:  gps_lat, gps_lon, gps_speed, gps_alt, gps_satellites, gps_fix
  Meta: timestamp, ecg_lead_off, label, session_id

Usage:
  1. Set the CURRENT_LABEL below to the activity being performed
  2. Run: python data_collection.py
  3. Perform the activity for the required duration
  4. Stop with CTRL+C, change label, repeat

Labels:
  - normal        : Soldier is fine — idle, sitting, standing, walking, patrolling
  - high_exertion : Sprinting, burpees, pushups, combat simulation
  - man_down      : Fall event — subject falls and stays still
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
from collections import deque

app = Flask(__name__)

# ==================== CONFIGURATION ====================
FILE_NAME = "soldier_data.csv"

# ============ CHANGE THESE BEFORE EACH SESSION ============
CURRENT_LABEL = "man_down"            # normal | high_exertion | man_down
SUBJECT_ID = "subject_01"          # Change for each volunteer
SESSION_ID = f"{CURRENT_LABEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ===========================================================

VALID_LABELS = ["normal", "high_exertion", "man_down"]

# Feature columns for the dataset
COLUMNS = [
    # Metadata
    "timestamp",
    "session_id",
    "subject_id",

    # ECG features
    "bpm",
    "hrv_sdnn",
    "hrv_rmssd",
    "ecg_lead_off",

    # IMU features — Accelerometer
    "smv",
    "dynamic_accel",
    "impact",
    "pitch",
    "roll",

    # IMU features — Gyroscope
    "gx",
    "gy",
    "gz",

    # IMU derived
    "movement_var",

    # Windowed features (computed server-side)
    "bpm_mean_10s",
    "bpm_std_10s",
    "dynamic_accel_mean_5s",
    "dynamic_accel_max_5s",
    "impact_max_5s",
    "pitch_mean_5s",
    "movement_var_mean_5s",
    "gyro_magnitude_mean_5s",

    # GPS (for context, NOT for ML)
    "gps_lat",
    "gps_lon",
    "gps_speed",
    "gps_alt",
    "gps_satellites",
    "gps_fix",

    # Label
    "label"
]

# ==================== WINDOWED FEATURE BUFFERS ====================
# At 1 Hz: 5s = 5 samples, 10s = 10 samples
WINDOW_5S = 5
WINDOW_10S = 10

bpm_buffer = deque(maxlen=WINDOW_10S)
dynamic_accel_buffer = deque(maxlen=WINDOW_5S)
impact_buffer = deque(maxlen=WINDOW_5S)
pitch_buffer = deque(maxlen=WINDOW_5S)
movement_var_buffer = deque(maxlen=WINDOW_5S)
gyro_mag_buffer = deque(maxlen=WINDOW_5S)

# ==================== STATISTICS ====================
sample_count = 0
start_time = None
last_print_time = 0

# ==================== INITIALIZE CSV ====================
if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(FILE_NAME, index=False)
    print(f"Created new dataset: {FILE_NAME}")
else:
    existing = pd.read_csv(FILE_NAME, nrows=0)
    missing_cols = set(COLUMNS) - set(existing.columns)
    if missing_cols:
        print(f"WARNING: Existing CSV is missing columns: {missing_cols}")
        print("Consider starting a new file or adding missing columns.")

# Validate label
if CURRENT_LABEL not in VALID_LABELS:
    print(f"ERROR: Invalid label '{CURRENT_LABEL}'")
    print(f"Valid labels: {VALID_LABELS}")
    sys.exit(1)

print("=" * 60)
print("  SOLDIER SURVIVAL DETECTION - DATA COLLECTION SERVER")
print("=" * 60)
print(f"  Label     : {CURRENT_LABEL}")
print(f"  Subject   : {SUBJECT_ID}")
print(f"  Session   : {SESSION_ID}")
print(f"  File      : {FILE_NAME}")
print(f"  Server    : http://0.0.0.0:5000/data")
print("=" * 60)
print("Press CTRL+C to stop\n")


def compute_windowed_features():
    """Compute rolling window features from buffers."""
    features = {}

    # BPM features (10s window)
    if len(bpm_buffer) >= 10:
        valid_bpm = [b for b in bpm_buffer if b > 0]
        if valid_bpm:
            features["bpm_mean_10s"] = round(np.mean(valid_bpm), 2)
            features["bpm_std_10s"] = round(np.std(valid_bpm), 2)
        else:
            features["bpm_mean_10s"] = 0
            features["bpm_std_10s"] = 0
    else:
        features["bpm_mean_10s"] = 0
        features["bpm_std_10s"] = 0

    # Dynamic acceleration features (5s window)
    if len(dynamic_accel_buffer) >= 10:
        features["dynamic_accel_mean_5s"] = round(np.mean(dynamic_accel_buffer), 4)
        features["dynamic_accel_max_5s"] = round(np.max(dynamic_accel_buffer), 4)
    else:
        features["dynamic_accel_mean_5s"] = 0
        features["dynamic_accel_max_5s"] = 0

    # Impact max (5s window)
    if len(impact_buffer) >= 10:
        features["impact_max_5s"] = round(np.max(impact_buffer), 4)
    else:
        features["impact_max_5s"] = 0

    # Pitch mean (5s window) — body orientation
    if len(pitch_buffer) >= 10:
        features["pitch_mean_5s"] = round(np.mean(pitch_buffer), 2)
    else:
        features["pitch_mean_5s"] = 0

    # Movement variance mean (5s window)
    if len(movement_var_buffer) >= 10:
        features["movement_var_mean_5s"] = round(np.mean(movement_var_buffer), 6)
    else:
        features["movement_var_mean_5s"] = 0

    # Gyroscope magnitude mean (5s window)
    if len(gyro_mag_buffer) >= 10:
        features["gyro_magnitude_mean_5s"] = round(np.mean(gyro_mag_buffer), 2)
    else:
        features["gyro_magnitude_mean_5s"] = 0

    return features


@app.route("/data", methods=["POST"])
def receive_data():
    global sample_count, start_time, last_print_time

    if start_time is None:
        start_time = time.time()

    data = request.json
    if not data:
        return "No data", 400

    timestamp = time.time()

    # Extract sensor values with defaults
    bpm = data.get("bpm", 0)
    hrv_sdnn = data.get("hrv_sdnn", 0)
    hrv_rmssd = data.get("hrv_rmssd", 0)
    ecg_lead_off = data.get("ecg_lead_off", 1)

    smv_val = data.get("smv", 1.0)
    dynamic_accel_val = data.get("dynamic_accel", 0)
    impact_val = data.get("impact", 0)
    pitch_val = data.get("pitch", 0)
    roll_val = data.get("roll", 0)
    gx = data.get("gx", 0)
    gy = data.get("gy", 0)
    gz = data.get("gz", 0)
    movement_var_val = data.get("movement_var", 0)

    gps_lat = data.get("gps_lat", 0)
    gps_lon = data.get("gps_lon", 0)
    gps_speed = data.get("gps_speed", 0)
    gps_alt = data.get("gps_alt", 0)
    gps_satellites = data.get("gps_satellites", 0)
    gps_fix = data.get("gps_fix", 0)

    # Update buffers for windowed features
    bpm_buffer.append(bpm)
    dynamic_accel_buffer.append(dynamic_accel_val)
    impact_buffer.append(impact_val)
    pitch_buffer.append(pitch_val)
    movement_var_buffer.append(movement_var_val)

    gyro_magnitude = (gx**2 + gy**2 + gz**2) ** 0.5
    gyro_mag_buffer.append(gyro_magnitude)

    # Compute windowed features
    windowed = compute_windowed_features()

    # Build the row
    row = {
        "timestamp": timestamp,
        "session_id": SESSION_ID,
        "subject_id": SUBJECT_ID,

        "bpm": round(bpm, 1),
        "hrv_sdnn": round(hrv_sdnn, 2),
        "hrv_rmssd": round(hrv_rmssd, 2),
        "ecg_lead_off": int(ecg_lead_off),

        "smv": round(smv_val, 4),
        "dynamic_accel": round(dynamic_accel_val, 4),
        "impact": round(impact_val, 4),
        "pitch": round(pitch_val, 2),
        "roll": round(roll_val, 2),

        "gx": round(gx, 2),
        "gy": round(gy, 2),
        "gz": round(gz, 2),

        "movement_var": round(movement_var_val, 6),

        "bpm_mean_10s": windowed["bpm_mean_10s"],
        "bpm_std_10s": windowed["bpm_std_10s"],
        "dynamic_accel_mean_5s": windowed["dynamic_accel_mean_5s"],
        "dynamic_accel_max_5s": windowed["dynamic_accel_max_5s"],
        "impact_max_5s": windowed["impact_max_5s"],
        "pitch_mean_5s": windowed["pitch_mean_5s"],
        "movement_var_mean_5s": windowed["movement_var_mean_5s"],
        "gyro_magnitude_mean_5s": windowed["gyro_magnitude_mean_5s"],

        "gps_lat": round(gps_lat, 6),
        "gps_lon": round(gps_lon, 6),
        "gps_speed": round(gps_speed, 2),
        "gps_alt": round(gps_alt, 1),
        "gps_satellites": int(gps_satellites),
        "gps_fix": int(gps_fix),

        "label": CURRENT_LABEL
    }

    # Append to CSV
    new_row = pd.DataFrame([row])
    new_row.to_csv(FILE_NAME, mode="a", header=False, index=False)

    sample_count += 1

    # Print status every 1 second
    if timestamp - last_print_time >= 1.0:
        elapsed = timestamp - start_time
        rate = sample_count / elapsed if elapsed > 0 else 0

        ecg_status = "OK" if ecg_lead_off == 0 else "LEAD OFF"
        gps_status = f"FIX({gps_satellites}sat)" if gps_fix else "NO FIX"

        print(f"[{CURRENT_LABEL:>14}] "
              f"BPM:{bpm:5.0f} | HRV:{hrv_sdnn:5.1f} | "
              f"Accel:{dynamic_accel_val:.3f} | Impact:{impact_val:.2f} | "
              f"Pitch:{pitch_val:+6.1f} | "
              f"ECG:{ecg_status} | GPS:{gps_status} | "
              f"N={sample_count} ({rate:.1f}/s)")

        last_print_time = timestamp

    return "OK", 200


@app.route("/status", methods=["GET"])
def status():
    """Health check endpoint."""
    elapsed = time.time() - start_time if start_time else 0
    return jsonify({
        "status": "running",
        "label": CURRENT_LABEL,
        "subject": SUBJECT_ID,
        "session": SESSION_ID,
        "samples": sample_count,
        "elapsed_seconds": round(elapsed, 1)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)