# COMPLETE PIPELINE — How to Run Everything
## AI-Based Soldier Survival & Emergency Detection System

---

## Architecture Overview

```
┌──────────┐     WiFi/HTTP      ┌─────────────────┐     WebSocket     ┌───────────┐
│  ESP32   │ ──── POST /data ──→│  Python Server   │ ────────────────→│ Dashboard │
│ (Sensors)│                    │  (Flask + ML)    │                  │ (Browser) │
│          │                    │                  │                  │           │
│ AD8232   │                    │ ┌──────────────┐ │                  │ Live BPM  │
│ MPU6050  │                    │ │ Trained Model│ │                  │ Charts    │
│ NEO-6M   │                    │ │ (joblib)     │ │                  │ GPS Map   │
└──────────┘                    │ └──────────────┘ │                  │ Alerts    │
                                └─────────────────┘                  └───────────┘
```

---

## Step-by-Step Instructions

### STEP 0: Install Dependencies
```bash
pip install -r requirements.txt
```

### STEP 1: Collect Data (TODAY)
Follow `DATA_COLLECTION_PROTOCOL.md` exactly.

```bash
# Set CURRENT_LABEL in data_collection.py, then:
python data_collection.py
```

Repeat for all 3 labels: `normal`, `high_exertion`, `man_down`

**Target:** 3000+ samples per class (12,000+ total)

### STEP 2: Train the Model
```bash
python model_training.py
```

This will:
- Clean your data (remove lead-off, startup artifacts, etc.)
- Train 4 models: Random Forest, XGBoost, SVM, Neural Network
- Compare them with cross-validation
- Auto-select the best one
- Save everything to `models/` folder
- Generate charts: confusion matrices, feature importance, ROC curves

### STEP 3: Run the Real-Time Dashboard
```bash
python realtime_dashboard.py
```

Then:
1. Open browser: **http://localhost:5000**
2. Update ESP32 code:
   - Set `SERVER_URL` to `http://YOUR_LAPTOP_IP:5000/data`
   - Set your WiFi credentials
3. Power on the ESP32 device
4. Watch the live dashboard!

---

## File Structure After Setup

```
Major Project/
├── arduino code/
├── templates/
│   └── dashboard.html           ← Live dashboard UI
├── models/                      ← Created after training
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── feature_names.joblib
│   ├── model_metadata.json
│   ├── training_report.txt
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── model_comparison.png
├── esp32_sensor_code.ino        ← Upload to ESP32
├── data_collection.py           ← Step 1: Collect data
├── model_training.py            ← Step 2: Train models
├── realtime_dashboard.py        ← Step 3: Run live system
├── soldier_data.csv             ← Collected sensor data
├── requirements.txt
├── DATA_COLLECTION_PROTOCOL.md
└── PIPELINE_README.md           ← This file
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Status Hero** | Current classification with color-coded alert level |
| **Confidence Bars** | ML model's confidence for each class |
| **Heart Rate** | Live BPM with HRV (SDNN, RMSSD) |
| **Body Motion** | Dynamic acceleration, impact, movement variance |
| **Body Orientation** | Pitch/roll angles + posture detection |
| **GPS Map** | Live position tracking with satellite count |
| **Real-Time Charts** | 60-second rolling history for all vitals |
| **Alert System** | Full-screen alert + audio alarm for man_down |
| **Event Log** | Timestamped history of status changes and alerts |

---

## ESP32 Quick Config

In `esp32_sensor_code.ino`, update these lines:
```cpp
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD  = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL     = "http://192.168.X.X:5000/data";  // Your laptop's IP
```

Find your laptop IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model training fails with "not enough data" | Collect more data — need 500+ samples per class |
| Dashboard shows "Waiting for data" | Check ESP32 WiFi connection and SERVER_URL |
| ECG shows "LEAD OFF" | Re-attach ECG electrodes, ensure good skin contact |
| GPS shows "No Fix" | Move outdoors, wait 1-2 minutes for satellite lock |
| Low model accuracy | Collect more diverse data, check DATA_COLLECTION_PROTOCOL.md |
| XGBoost not available | Run: `pip install xgboost` |
