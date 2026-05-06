import sys
from realtime_dashboard import load_model, classify_state, check_alerts, compute_windowed_features, bpm_buffer, dynamic_accel_buffer, impact_buffer, pitch_buffer, movement_var_buffer, gyro_mag_buffer

try:
    sensor_data = {
        "bpm": 30, "hrv_sdnn": 10, "hrv_rmssd": 5, "dynamic_accel": 0.02, "impact": 0.0,
        "pitch": 90, "roll": 90, "gx": 0, "gy": 0, "gz": 0, "movement_var": 0.005,
        "ecg_lead_off": 0, "gps_lat": 12.9716, "gps_lon": 77.5946, "gps_speed": 0.0,
        "gps_alt": 900, "gps_satellites": 8, "gps_fix": 1
    }
    bpm_buffer.append(sensor_data["bpm"])
    dynamic_accel_buffer.append(sensor_data["dynamic_accel"])
    impact_buffer.append(sensor_data["impact"])
    pitch_buffer.append(sensor_data["pitch"])
    movement_var_buffer.append(sensor_data["movement_var"])
    gyro_mag_buffer.append(0)
    windowed = compute_windowed_features()
    status, confidences = classify_state(sensor_data, windowed)
    print("Classify OK:", status, confidences)
    
    for i in range(4):
        alert = check_alerts(status, confidences)
        print("Alert", i, ":", alert)
except Exception as e:
    import traceback
    traceback.print_exc()
