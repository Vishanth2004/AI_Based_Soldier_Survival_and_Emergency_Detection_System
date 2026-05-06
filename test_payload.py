import json
import realtime_dashboard as rd

app = rd.app
app.testing = True
client = app.test_client()

data = {
    "bpm": 30.0,
    "hrv_sdnn": 10.0,
    "hrv_rmssd": 5.0,
    "dynamic_accel": 0.02,
    "impact": 0.0,
    "pitch": 90.0,
    "roll": 90.0,
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
    "movement_var": 0.005,
    "gps_lat": 12.9716,
    "gps_lon": 77.5946,
    "gps_speed": 0.0,
    "gps_alt": 900,
    "gps_satellites": 8,
    "gps_fix": 1,
    "ecg_lead_off": 0
}

# The problem happens when socketio tries to emit.
# In a test client, socketio emit might not fail if it's mocked, but flask_socketio will serialize.
for i in range(5):
    print(f"--- POST {i} ---")
    response = client.post("/data", json=data)
    print("Status:", response.status_code)
    if response.status_code != 200:
        print(response.data)

