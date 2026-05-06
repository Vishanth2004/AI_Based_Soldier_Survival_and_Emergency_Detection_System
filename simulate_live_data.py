import time
import requests
import numpy as np

# Configuration
URL = "http://localhost:5001/data"
SEND_INTERVAL = 1.0  # Send 1 sample per second (1 Hz)

def simulate_data(state):
    """Generates a single frame of data based on the current state."""
    if state == 'normal':
        return {
            "bpm": np.random.normal(70, 5),
            "hrv_sdnn": np.random.normal(50, 10),
            "hrv_rmssd": np.random.normal(35, 8),
            "dynamic_accel": np.random.normal(0.2, 0.1),
            "impact": np.random.exponential(0.05),
            "pitch": np.random.normal(10, 5),
            "roll": np.random.normal(0, 5),
            "gx": np.random.normal(0, 5),
            "gy": np.random.normal(0, 5),
            "gz": np.random.normal(0, 5),
            "movement_var": np.random.normal(0.05, 0.02),
            "ecg_lead_off": 0,
            "gps_lat": 12.9716 + np.random.normal(0, 0.0001),
            "gps_lon": 77.5946 + np.random.normal(0, 0.0001),
            "gps_speed": np.random.normal(1.2, 0.2),
            "gps_alt": 900 + np.random.normal(0, 2),
            "gps_satellites": 8,
            "gps_fix": 1
        }
    elif state == 'high_exertion':
        return {
            "bpm": np.random.normal(160, 10),
            "hrv_sdnn": np.random.normal(25, 5),
            "hrv_rmssd": np.random.normal(15, 5),
            "dynamic_accel": np.random.normal(2.0, 0.5),
            "impact": np.random.exponential(0.5),
            "pitch": np.random.normal(20, 10),
            "roll": np.random.normal(0, 10),
            "gx": np.random.normal(0, 50),
            "gy": np.random.normal(0, 50),
            "gz": np.random.normal(0, 50),
            "movement_var": np.random.normal(0.5, 0.1),
            "ecg_lead_off": 0,
            "gps_lat": 12.9716 + np.random.normal(0, 0.0005),
            "gps_lon": 77.5946 + np.random.normal(0, 0.0005),
            "gps_speed": np.random.normal(5.0, 1.0),
            "gps_alt": 900 + np.random.normal(0, 5),
            "gps_satellites": 8,
            "gps_fix": 1
        }
    elif state == 'man_down':
        return {
            "bpm": np.random.normal(30, 5),
            "hrv_sdnn": np.random.normal(10, 5),
            "hrv_rmssd": np.random.normal(5, 2),
            "dynamic_accel": np.random.normal(0.02, 0.01),
            "impact": 0.0,
            "pitch": np.random.normal(90, 5),
            "roll": np.random.normal(90, 10),
            "gx": np.random.normal(0, 1),
            "gy": np.random.normal(0, 1),
            "gz": np.random.normal(0, 1),
            "movement_var": np.random.normal(0.005, 0.002),
            "ecg_lead_off": 0,
            "gps_lat": 12.9716,
            "gps_lon": 77.5946,
            "gps_speed": 0.0,
            "gps_alt": 900,
            "gps_satellites": 8,
            "gps_fix": 1
        }

def main():
    print("========================================")
    print("  ESP32 LIVE DATA SIMULATOR")
    print("========================================")
    print(f"Target URL: {URL}")
    print("Press CTRL+C to switch states or stop.")
    print("----------------------------------------")
    
    states = ['normal', 'high_exertion', 'man_down']
    
    for state in states:
        print(f"\n>>> Simulating state: {state.upper()} <<<")
        print("Sending data... (Auto-switching state in 15 seconds)")
        
        # Run each state for 15 seconds
        end_time = time.time() + 15
        
        try:
            while time.time() < end_time:
                data = simulate_data(state)
                
                # Clip values just like real hardware constraints
                data['bpm'] = max(0, min(220, data['bpm']))
                data['dynamic_accel'] = max(0, data['dynamic_accel'])
                data['impact'] = max(0, data['impact'])
                data['movement_var'] = max(0, data['movement_var'])
                
                try:
                    response = requests.post(URL, json=data, timeout=2.0)
                    if response.status_code == 200:
                        print(f"[{state}] Sent -> BPM: {data['bpm']:3.0f} | Pitch: {data['pitch']:5.1f} | Accel: {data['dynamic_accel']:4.2f}")
                    else:
                        print(f"Error: Server returned {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Connection error: Make sure realtime_dashboard.py is running on port 5001!")
                
                time.sleep(SEND_INTERVAL)
        except KeyboardInterrupt:
            print("\nManually switching state...")
            time.sleep(1)
            continue
            
        print("\nSwitching state...")
        time.sleep(1)
            
    print("\nSimulation complete. All states tested.")

if __name__ == "__main__":
    main()
