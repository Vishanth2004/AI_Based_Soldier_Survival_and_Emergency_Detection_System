"""
AI-Based Soldier Survival & Emergency Detection System
REAL-TIME DASHBOARD SERVER

Receives live sensor data from ESP32, runs ML classification,
and serves a real-time web dashboard with WebSocket updates.

Architecture:
  ESP32 → HTTP POST /data → Flask Server → ML Model → WebSocket → Dashboard

Usage:
  1. Train the model first: python model_training.py
  2. Run: python realtime_dashboard.py
  3. Open browser: http://localhost:5000
  4. Power on ESP32 (pointed to this laptop's IP:5000)

The dashboard shows:
  - Current soldier status (Normal / High Exertion / Man Down)
  - Live vitals: BPM, HRV, body orientation
  - Motion metrics: acceleration, impact, movement
  - Real-time charts (last 60 seconds)
  - GPS map with live position
  - Alert system with audio for critical states
  - Classification confidence breakdown
  - Session log / event history
"""

import os
import sys
import time
import json
import logging
import numpy as np
from datetime import datetime
from collections import deque
from threading import Lock

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO

import joblib

# ==================== CONFIGURATION ====================
MODEL_DIR = "models"
HOST = "0.0.0.0"
PORT = 5001
DASHBOARD_UPDATE_RATE = 4  # Hz — how often to push updates to dashboard

# Alert thresholds
ALERT_CONSECUTIVE_THRESHOLD = 3  # Need N consecutive man_down to trigger alert
ALERT_COOLDOWN_SECONDS = 10       # Don't re-alert within this window

# Temporal smoothing: majority vote over last N predictions
# Prevents jittery single-sample misclassifications
SMOOTHING_WINDOW = 3  # vote over last 3 predictions (~3 s at 1 Hz)

# Windowed feature buffer sizes (matching data_collection.py)
WINDOW_5S = 5    # At 1 Hz: 5s = 5 samples
WINDOW_10S = 10  # At 1 Hz: 10s = 10 samples

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ==================== FLASK APP ====================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "soldier-survival-system-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ==================== LOAD ML MODEL ====================
def load_model():
    """Load the trained ML model and preprocessing artifacts."""
    required_files = [
        "best_model.joblib",
        "scaler.joblib",
        "label_encoder.joblib",
        "feature_names.joblib",
    ]

    for f in required_files:
        path = os.path.join(MODEL_DIR, f)
        if not os.path.exists(path):
            log.error(f"Missing model file: {path}")
            log.error("Run 'python model_training.py' first!")
            sys.exit(1)

    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

    # Load metadata
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    log.info(f"Model loaded: {metadata.get('best_model', 'Unknown')}")
    log.info(f"Accuracy: {metadata.get('accuracy', 'N/A')}")
    log.info(f"Labels: {list(label_encoder.classes_)}")

    return model, scaler, label_encoder, feature_names, metadata

# Load model at startup
ml_model, ml_scaler, ml_label_encoder, ml_feature_names, ml_metadata = load_model()

# ==================== JSON UTILITIES ====================
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    elif type(obj).__module__ == 'numpy':
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)
    return obj

# ==================== STATE ====================
data_lock = Lock()
last_sim_time = 0

# Current state
current_state = {
    "status": "waiting",        # waiting | normal | high_exertion | man_down
    "confidence": {},
    "bpm": 0,
    "hrv_sdnn": 0,
    "hrv_rmssd": 0,
    "dynamic_accel": 0,
    "impact": 0,
    "pitch": 0,
    "roll": 0,
    "gx": 0,
    "gy": 0,
    "gz": 0,
    "movement_var": 0,
    "gps_lat": 0,
    "gps_lon": 0,
    "gps_speed": 0,
    "gps_alt": 0,
    "gps_satellites": 0,
    "gps_fix": False,
    "ecg_lead_off": True,
    "last_update": 0,
    "samples_received": 0,
    "uptime": 0,
}

# Windowed buffers for computing rolling features (same as data_collection.py)
bpm_buffer = deque(maxlen=WINDOW_10S)
dynamic_accel_buffer = deque(maxlen=WINDOW_5S)
impact_buffer = deque(maxlen=WINDOW_5S)
pitch_buffer = deque(maxlen=WINDOW_5S)
movement_var_buffer = deque(maxlen=WINDOW_5S)
gyro_mag_buffer = deque(maxlen=WINDOW_5S)

# History for charts (last 60 seconds at ~1 Hz = 60 points)
CHART_HISTORY = 60
history = {
    "timestamps": deque(maxlen=CHART_HISTORY),
    "bpm": deque(maxlen=CHART_HISTORY),
    "hrv_sdnn": deque(maxlen=CHART_HISTORY),
    "dynamic_accel": deque(maxlen=CHART_HISTORY),
    "impact": deque(maxlen=CHART_HISTORY),
    "pitch": deque(maxlen=CHART_HISTORY),
    "movement_var": deque(maxlen=CHART_HISTORY),
    "status": deque(maxlen=CHART_HISTORY),
}

# Prediction smoothing buffer
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

# Alert tracking
alert_state = {
    "consecutive_critical": 0,
    "last_alert_time": 0,
    "alert_active": False,
    "alert_type": None,
}

# Event log
event_log = deque(maxlen=100)
start_time = time.time()


# ==================== WINDOWED FEATURES ====================
def compute_windowed_features():
    """Compute rolling window features matching data_collection.py."""
    features = {}

    if len(bpm_buffer) >= 4:
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

    if len(dynamic_accel_buffer) >= 4:
        features["dynamic_accel_mean_5s"] = round(np.mean(dynamic_accel_buffer), 4)
        features["dynamic_accel_max_5s"] = round(np.max(dynamic_accel_buffer), 4)
    else:
        features["dynamic_accel_mean_5s"] = 0
        features["dynamic_accel_max_5s"] = 0

    if len(impact_buffer) >= 4:
        features["impact_max_5s"] = round(np.max(impact_buffer), 4)
    else:
        features["impact_max_5s"] = 0

    if len(pitch_buffer) >= 4:
        features["pitch_mean_5s"] = round(np.mean(pitch_buffer), 2)
    else:
        features["pitch_mean_5s"] = 0

    if len(movement_var_buffer) >= 4:
        features["movement_var_mean_5s"] = round(np.mean(movement_var_buffer), 6)
    else:
        features["movement_var_mean_5s"] = 0

    if len(gyro_mag_buffer) >= 4:
        features["gyro_magnitude_mean_5s"] = round(np.mean(gyro_mag_buffer), 2)
    else:
        features["gyro_magnitude_mean_5s"] = 0

    return features


# ==================== ML INFERENCE ====================
def classify_state(sensor_data, windowed_features):
    """Run the ML model on incoming sensor data and return classification.

    Uses a majority-vote smoothing window to stabilise predictions
    and avoid single-sample jitter (e.g. one pushup rep classified as
    man_down while the overall pattern is clearly high_exertion).
    """
    try:
        # Build feature vector in the exact order the model expects
        import pandas as pd
        combined = {**sensor_data, **windowed_features}
        feature_dict = {feat: [combined.get(feat, 0)] for feat in ml_feature_names}
        feature_df = pd.DataFrame(feature_dict, columns=list(ml_feature_names))

        # Scale (DataFrame preserves feature names — no sklearn warning)
        feature_scaled = ml_scaler.transform(feature_df)

        # ---- raw model prediction ----
        raw_prediction = ml_model.predict(feature_scaled)[0]
        raw_label = ml_label_encoder.inverse_transform([raw_prediction])[0]

        # Get probabilities
        confidences = {}
        if hasattr(ml_model, "predict_proba"):
            proba = ml_model.predict_proba(feature_scaled)[0]
            for i, class_name in enumerate(ml_label_encoder.classes_):
                confidences[class_name] = round(float(proba[i]) * 100, 1)

        # ---- temporal majority-vote smoothing ----
        prediction_buffer.append(raw_label)
        if len(prediction_buffer) >= SMOOTHING_WINDOW:
            from collections import Counter
            vote_counts = Counter(prediction_buffer)
            smoothed_label = vote_counts.most_common(1)[0][0]
        else:
            smoothed_label = raw_label

        # Debug: log every ~5 s (every 5th sample at 1 Hz)
        sample_count = current_state.get("samples_received", 0)
        if sample_count % 5 == 0:
            top2 = sorted(confidences.items(), key=lambda x: -x[1])[:2]
            top2_str = ", ".join(f"{k}={v}%" for k, v in top2)
            log.info(
                f"[CLASS] raw={raw_label}  smoothed={smoothed_label}  "
                f"conf=[{top2_str}]  "
                f"bpm={sensor_data['bpm']:.0f}  pitch={sensor_data['pitch']:.1f}  "
                f"mvVar={sensor_data['movement_var']:.4f}  "
                f"dynAcc={sensor_data['dynamic_accel']:.3f}"
            )

        return smoothed_label, confidences

    except Exception as e:
        log.error(f"Classification error: {e}")
        return "unknown", {}


# ==================== ALERT LOGIC ====================
def check_alerts(status, confidences):
    """Check if an alert should be triggered."""
    global alert_state

    is_critical = status == "man_down"
    now = time.time()

    if is_critical:
        alert_state["consecutive_critical"] += 1
    else:
        alert_state["consecutive_critical"] = 0
        if alert_state["alert_active"]:
            # Clear alert after sustained normal state
            alert_state["alert_active"] = False
            alert_state["alert_type"] = None
            add_event("ALERT CLEARED", f"Status returned to: {status}")
            return {"action": "clear_alert"}

    # Trigger alert if enough consecutive critical readings
    if (alert_state["consecutive_critical"] >= ALERT_CONSECUTIVE_THRESHOLD
            and (now - alert_state["last_alert_time"]) > ALERT_COOLDOWN_SECONDS):

        alert_state["alert_active"] = True
        alert_state["alert_type"] = status
        alert_state["last_alert_time"] = now

        priority = "CRITICAL" if status == "man_down" else "WARNING"
        confidence = confidences.get(status, 0)
        add_event(f"🚨 {priority} ALERT", f"{status.upper()} detected ({confidence}% confidence)")

        return {
            "action": "trigger_alert",
            "type": status,
            "priority": priority,
            "confidence": confidence,
        }

    return None


def add_event(title, description):
    """Add event to the log."""
    event_log.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "title": title,
        "description": description,
    })


# ==================== ROUTES ====================

@app.route("/")
def dashboard():
    """Serve the dashboard page."""
    return render_template("dashboard.html", metadata=ml_metadata)


@app.route("/data", methods=["POST"])
def receive_data():
    """Receive sensor data from ESP32 (same endpoint as data_collection.py)."""
    global current_state, last_sim_time

    data = request.json
    if not data:
        return "No data", 400

    now = time.time()
    is_simulator = (request.remote_addr == "127.0.0.1")

    if is_simulator:
        last_sim_time = now
    else:
        # Ignore ESP32 data if the simulator has sent data within the last 5 seconds
        if now - last_sim_time < 5.0:
            return "Ignored (Simulator Active)", 200

    with data_lock:
        # Extract sensor values
        sensor_data = {
            "bpm": data.get("bpm", 0),
            "hrv_sdnn": data.get("hrv_sdnn", 0),
            "hrv_rmssd": data.get("hrv_rmssd", 0),
            "dynamic_accel": data.get("dynamic_accel", 0),
            "impact": data.get("impact", 0),
            "pitch": data.get("pitch", 0),
            "roll": data.get("roll", 0),
            "gx": data.get("gx", 0),
            "gy": data.get("gy", 0),
            "gz": data.get("gz", 0),
            "movement_var": data.get("movement_var", 0),
        }

        gps_data = {
            "gps_lat": data.get("gps_lat", 0),
            "gps_lon": data.get("gps_lon", 0),
            "gps_speed": data.get("gps_speed", 0),
            "gps_alt": data.get("gps_alt", 0),
            "gps_satellites": data.get("gps_satellites", 0),
            "gps_fix": bool(data.get("gps_fix", 0)),
        }

        ecg_lead_off = bool(data.get("ecg_lead_off", 1))

        # Update windowed buffers
        bpm_buffer.append(sensor_data["bpm"])
        dynamic_accel_buffer.append(sensor_data["dynamic_accel"])
        impact_buffer.append(sensor_data["impact"])
        pitch_buffer.append(sensor_data["pitch"])
        movement_var_buffer.append(sensor_data["movement_var"])
        gyro_mag = (sensor_data["gx"]**2 + sensor_data["gy"]**2 + sensor_data["gz"]**2) ** 0.5
        gyro_mag_buffer.append(gyro_mag)

        # Compute windowed features
        windowed = compute_windowed_features()

        # Classify!
        status, confidences = classify_state(sensor_data, windowed)

        # Check alerts
        alert = check_alerts(status, confidences)

        # Update current state
        current_state.update(sensor_data)
        current_state.update(gps_data)
        current_state.update(windowed)
        current_state["status"] = status
        current_state["confidence"] = confidences
        current_state["ecg_lead_off"] = ecg_lead_off
        current_state["last_update"] = now
        current_state["samples_received"] = current_state.get("samples_received", 0) + 1
        current_state["uptime"] = round(now - start_time, 1)

        # Update history for charts
        history["timestamps"].append(now)
        history["bpm"].append(sensor_data["bpm"])
        history["hrv_sdnn"].append(sensor_data["hrv_sdnn"])
        history["dynamic_accel"].append(sensor_data["dynamic_accel"])
        history["impact"].append(sensor_data["impact"])
        history["pitch"].append(sensor_data["pitch"])
        history["movement_var"].append(sensor_data["movement_var"])
        history["status"].append(status)

        # Push to dashboard via WebSocket
        dashboard_payload = {
            **current_state,
            "alert": alert,
            "confidence": confidences,
        }
        socketio.emit("sensor_update", make_json_safe(dashboard_payload))

    return "OK", 200


@app.route("/api/state", methods=["GET"])
def get_state():
    """REST endpoint for current state."""
    with data_lock:
        return jsonify(make_json_safe(current_state))


@app.route("/api/history", methods=["GET"])
def get_history():
    """REST endpoint for chart history."""
    with data_lock:
        return jsonify(make_json_safe({
            key: list(val) for key, val in history.items()
        }))


@app.route("/api/events", methods=["GET"])
def get_events():
    """REST endpoint for event log."""
    return jsonify(make_json_safe(list(event_log)))


@app.route("/api/model", methods=["GET"])
def get_model_info():
    """REST endpoint for model metadata."""
    return jsonify(make_json_safe(ml_metadata))


# ==================== WEBSOCKET EVENTS ====================

@socketio.on("connect")
def handle_connect():
    log.info("Dashboard client connected")
    add_event("Dashboard Connected", "Web client connected to server")
    # Send current state immediately
    socketio.emit("sensor_update", make_json_safe(current_state))
    socketio.emit("event_log", make_json_safe(list(event_log)))


@socketio.on("disconnect")
def handle_disconnect():
    log.info("Dashboard client disconnected")


@socketio.on("request_history")
def handle_history_request():
    """Client requests chart history (on page load)."""
    with data_lock:
        socketio.emit("history_data", make_json_safe({
            key: list(val) for key, val in history.items()
        }))


@socketio.on("clear_alert")
def handle_clear_alert():
    """Client manually clears an alert."""
    global alert_state
    alert_state["alert_active"] = False
    alert_state["alert_type"] = None
    alert_state["consecutive_critical"] = 0
    add_event("Alert Dismissed", "Manually cleared by operator")
    socketio.emit("alert_cleared")


# ==================== STARTUP ====================

def main():
    print()
    print("█" * 70)
    print("█  SOLDIER SURVIVAL DETECTION — REAL-TIME DASHBOARD")
    print("█" * 70)
    print(f"  Model:     {ml_metadata.get('best_model', 'Unknown')}")
    print(f"  Accuracy:  {ml_metadata.get('accuracy', 'N/A')}")
    print(f"  Features:  {ml_metadata.get('n_features', len(ml_feature_names))}")
    print(f"  Classes:   {ml_metadata.get('labels', list(ml_label_encoder.classes_))}")
    print(f"  Server:    http://0.0.0.0:{PORT}")
    print(f"  Dashboard: http://localhost:{PORT}")
    print()
    print("  Waiting for ESP32 data on POST /data ...")
    print("  Open the dashboard URL in your browser.")
    print("█" * 70)
    print()

    add_event("Server Started", f"Model: {ml_metadata.get('best_model', 'Unknown')}")

    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
