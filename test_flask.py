import sys
from realtime_dashboard import app

app.config['TESTING'] = True
client = app.test_client()

data = {
    "bpm": 30, "hrv_sdnn": 10, "hrv_rmssd": 5, "dynamic_accel": 0.02, "impact": 0.0,
    "pitch": 90, "roll": 90, "gx": 0, "gy": 0, "gz": 0, "movement_var": 0.005,
    "ecg_lead_off": 0, "gps_lat": 12.9716, "gps_lon": 77.5946, "gps_speed": 0.0,
    "gps_alt": 900, "gps_satellites": 8, "gps_fix": 1
}

for i in range(4):
    response = client.post('/data', json=data)
    print("Response", i, ":", response.status_code)
    if response.status_code != 200:
        print(response.data.decode('utf-8'))
