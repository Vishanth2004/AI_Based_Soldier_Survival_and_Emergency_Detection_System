# Architecture Document

**Project Title:** AI-Based Soldier Survival & Emergency Detection System  
**Version:** 1.0  
**Date:** March 5, 2026

---

## Table of Contents

1. [Architecture Selection: Event-Driven Architecture](#1-architecture-selection-event-driven-architecture-eda)
2. [Three-Tier Architecture](#2-three-tier-architecture)
3. [Hardware Architecture](#3-hardware-architecture)
4. [Software Architecture](#4-software-architecture)
5. [Data Architecture](#5-data-architecture)
6. [Communication Protocols](#6-communication-protocols)
7. [Technology Stack Summary](#7-technology-stack-summary)
8. [Use Case Diagram](#8-use-case-diagram)
9. [Class Diagram](#9-class-diagram)
10. [Data Flow Diagram](#10-data-flow-diagram)
11. [Component Diagram](#11-component-diagram)
12. [Sequence Diagram](#12-sequence-diagram)
13. [Deployment Diagram](#13-deployment-diagram)
14. [ER Diagram](#14-er-diagram)
15. [Schema Design](#15-schema-design)
16. [Data Exchange Contract](#16-data-exchange-contract)

---

## 1. Architecture Selection: Event-Driven Architecture (EDA)

For this AI-Based Soldier Survival & Emergency Detection System, we have selected **Event-Driven Architecture (EDA)** as the primary architectural pattern.

### Why Event-Driven Architecture?

| Factor | Justification |
|--------|---------------|
| **Real-Time Requirements** | Soldier state changes (Normal → High Exertion → Man Down) must be detected and communicated instantly. EDA's publish-subscribe model enables immediate event propagation. |
| **Loose Coupling** | Sensors, processing layer, and dashboard operate independently. The ESP32 doesn't need to know about the ML model; it just publishes sensor events. |
| **Asynchronous Processing** | Sensor data arrives continuously at 1 Hz. EDA allows the system to process events asynchronously without blocking. |
| **Scalability** | EDA naturally supports multiple soldiers (event producers) and multiple dashboards (event consumers) without architectural changes. |
| **Fault Tolerance** | If the dashboard disconnects, the server continues processing; when it reconnects, it receives the latest state. |
| **Alert-Driven Actions** | Critical events (Man Down) trigger immediate alerts — a fundamental EDA use case. |

### Architecture Comparison

| Architecture | Suitability | Reason |
|--------------|-------------|--------|
| **Event-Driven** ✓ | **Best Fit** | Real-time sensor events, asynchronous processing, alert broadcasting |
| Microservices | Overkill | Single-node deployment; no need for distributed service mesh |
| Serverless | Not Suitable | Requires persistent connections (WebSocket), continuous processing |
| Monolithic | Partial Fit | Currently implemented as monolith, but follows EDA patterns internally |

### Event Types in the System

| Event Type | Producer | Consumer | Payload | Frequency |
|------------|----------|----------|---------|-----------|
| `sensor_data` | ESP32 | Flask Server | JSON with 19 sensor readings | 1 Hz |
| `classification_result` | ML Pipeline | Dashboard | State, confidence, vitals | 1 Hz |
| `status_change` | Classification Engine | Dashboard, Alert System | Previous state, new state | On change |
| `alert_triggered` | Alert Manager | Dashboard | Alert type, timestamp, GPS | On critical event |
| `alert_cleared` | Dashboard (user) | Server | Alert ID, clear timestamp | On user action |

### Architecture Diagram

```mermaid
graph TB
    subgraph "EVENT PRODUCERS"
        ESP32[ESP32 Wearable<br/>Sensors]
    end
    
    subgraph "EVENT BROKER / MEDIATOR"
        subgraph "Flask Server + SocketIO"
            Queue[Event Queue<br/>Deque Buffer]
            Processor[Event Processing<br/>ML Classifier]
            Router[Event Router<br/>WebSocket Emit]
            Store[Event Store<br/>CSV/JSON]
        end
    end
    
    subgraph "EVENT CONSUMERS"
        Dashboard[Web Dashboard]
        AlertHandler[Alert Handler<br/>Notification]
    end
    
    ESP32 -->|HTTP POST<br/>Sensor Events| Queue
    Queue --> Processor
    Processor --> Router
    Processor --> Store
    Router -->|WebSocket Events| Dashboard
    Router -->|Alert Events| AlertHandler
    AlertHandler --> Dashboard
```

---

## 2. Three-Tier Architecture

| Tier | Component | Technology | Description |
|------|-----------|------------|-------------|
| **Hardware / Sensor Layer** | Wearable Device | ESP32 + AD8232 + MPU6050 + NEO-6M | Collects ECG, motion, and GPS data; performs on-device signal processing; transmits JSON over Wi-Fi |
| **Application / Server Layer** | Flask Server | Python, Flask, Flask-SocketIO, scikit-learn, XGBoost, joblib | Receives sensor data, computes windowed features, runs ML inference, manages alerts, pushes updates via WebSocket |
| **Presentation Layer** | Web Dashboard | HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js, Leaflet.js, Socket.IO | Displays real-time status, charts, GPS map, alerts, and event log |

---

## 3. Hardware Architecture

### 3.1 Sensor Specifications

| Sensor | Model | Interface | Placement | Sampling Rate | Purpose |
|--------|-------|-----------|-----------|---------------|---------|
| ECG | AD8232 | Analog (GPIO34) | Chest (3-electrode) | ~166 Hz (internal) | Heart rate, HRV |
| IMU | MPU6050 | I2C (GPIO21/22) | Chest/Sternum | Per-cycle (~166 Hz) | Motion, orientation, fall detection |
| GPS | NEO-6M | UART2 (GPIO16/17) | Shoulder | 1 Hz (GPS standard) | Location tracking |
| MCU | ESP32 | — | Waist belt | — | Central processing + Wi-Fi transmission |

### 3.2 Wiring Diagram

| Pin | Connection |
|-----|------------|
| GPIO34 | AD8232 OUTPUT (ECG analog) |
| GPIO32 | AD8232 LO+ (lead-off detect) |
| GPIO33 | AD8232 LO- (lead-off detect) |
| GPIO21 | MPU6050 SDA (I2C data) |
| GPIO22 | MPU6050 SCL (I2C clock) |
| GPIO16 | NEO-6M TX → ESP32 RX2 |
| GPIO17 | NEO-6M RX ← ESP32 TX2 |

### 3.3 On-Device Processing Pipeline

```mermaid
graph LR
    subgraph "ESP32 On-Device Processing"
        ECG[Raw ECG 166Hz] --> AR[Artifact Rejection]
        AR --> BT[Baseline Tracking]
        BT --> TD[Threshold Detection]
        TD --> RP[R-Peak]
        RP --> BPM[BPM Median Filter]
        BPM --> HRV[HRV SDNN/RMSSD]
        
        IMU[Raw IMU] --> GC[G-Conversion]
        GC --> SMV[SMV]
        SMV --> DA[Dynamic Accel]
        DA --> Impact[Impact/Jerk]
        Impact --> PR[Pitch/Roll]
        PR --> MV[Movement Variance]
        
        GPS[Raw NMEA] --> Parse[Parse GPGGA/GPRMC]
        Parse --> Coords[Lat/Lon/Speed/Alt]
    end
    
    HRV --> JSON[JSON Payload]
    MV --> JSON
    Coords --> JSON
    JSON -->|HTTP POST 1Hz| Server[Flask Server]
```

---

## 4. Software Architecture

### 4.1 Component Summary

| Component | File | Role |
|-----------|------|------|
| ESP32 Firmware | `esp32_sensor_code.ino` | Sensor reading, signal processing, Wi-Fi transmission |
| Data Collection Server | `data_collection.py` | Labeled data collection for training |
| Model Training Pipeline | `model_training.py` | Data cleaning, training 4 models, evaluation, model selection |
| Real-Time Dashboard Server | `realtime_dashboard.py` | Live inference, alerting, WebSocket push |
| Dashboard UI | `templates/dashboard.html` | Single-page real-time visualization |
| Dataset Analysis | `analyze_dataset.py` | Statistical analysis of collected data |

### 4.2 ML Pipeline Architecture

```mermaid
graph TB
    CSV[(soldier_data.csv)] --> Clean[Data Cleaning<br/>Remove lead-off, artifacts, NaN]
    Clean --> Extract[Feature Extraction<br/>19 features]
    Extract --> Split[Train/Test Split<br/>80/20 stratified]
    Split --> Scale[StandardScaler<br/>Fit on train only]
    
    Scale --> RF[Random Forest]
    Scale --> XGB[XGBoost]
    Scale --> SVM[SVM]
    Scale --> MLP[Neural Network]
    
    RF --> CV[5-Fold Cross-Validation]
    XGB --> CV
    SVM --> CV
    MLP --> CV
    
    CV --> Select[Select Best<br/>by weighted F1]
    Select --> Save[(Save to models/<br/>.joblib files)]
```

### 4.3 Real-Time Inference Flow

```mermaid
graph TB
    ESP32[ESP32 POST /data] --> Handler[Flask Route Handler]
    Handler --> Buffer[Update windowed buffers<br/>bpm, accel, impact, pitch, gyro]
    Buffer --> Compute[Compute 8 windowed features<br/>rolling 5s/10s]
    Compute --> Vector[Build 19-feature vector]
    Vector --> Scale[StandardScaler transform]
    Scale --> Predict[XGBoost.predict]
    Predict --> Smooth[Majority-vote smoothing<br/>last 5 predictions]
    Smooth --> Alert{3 consecutive<br/>man_down?}
    Alert -->|Yes| Critical[CRITICAL Alert]
    Alert -->|No| WS[WebSocket emit]
    Critical --> WS
    WS --> Dashboard[Dashboard update]
```

---

## 5. Data Architecture

### 5.1 Sensor Data Schema (JSON from ESP32)

```json
{
  "bpm": 72.5,
  "hrv_sdnn": 45.2,
  "hrv_rmssd": 38.1,
  "smv": 1.02,
  "dynamic_accel": 0.02,
  "impact": 0.15,
  "pitch": 5.3,
  "roll": -2.1,
  "gx": 1.2,
  "gy": -0.5,
  "gz": 0.8,
  "movement_var": 0.0003,
  "gps_lat": 28.6139,
  "gps_lon": 77.2090,
  "gps_speed": 0.0,
  "gps_alt": 216.0,
  "gps_satellites": 7,
  "gps_fix": 1,
  "ecg_lead_off": 0
}
```

### 5.2 Training Dataset Schema (soldier_data.csv)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | Unix timestamp |
| session_id | string | Unique session identifier |
| subject_id | string | Volunteer identifier |
| bpm ... movement_var | float | 11 raw sensor features |
| bpm_mean_10s ... gyro_magnitude_mean_5s | float | 8 windowed features |
| gps_lat ... gps_fix | float/bool | GPS context data |
| label | string | Ground truth class label |

### 5.3 Model Artifacts (models/ directory)

| File | Format | Description |
|------|--------|-------------|
| best_model.joblib | Joblib | Trained XGBoost classifier |
| scaler.joblib | Joblib | StandardScaler fitted on training data |
| label_encoder.joblib | Joblib | LabelEncoder for class names |
| feature_names.joblib | Joblib | Ordered list of 19 feature names |
| model_metadata.json | JSON | Model name, accuracy, F1, training timestamp |
| training_report.txt | Text | Full comparison report of all 4 models |

---

## 6. Communication Protocols

| Link | Protocol | Format | Frequency |
|------|----------|--------|-----------|
| ESP32 → Server | HTTP POST (Wi-Fi) | JSON | 1 Hz (1000ms) |
| Server → Dashboard | WebSocket (Socket.IO) | JSON | ~1 Hz |
| Dashboard → Server | WebSocket events | — | On-demand (clear alert, request history) |
| Server REST APIs | HTTP GET | JSON | On-demand (`/api/state`, `/api/history`, `/api/events`, `/api/model`) |

---

## 7. Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| Microcontroller | ESP32 (Arduino C++) |
| ECG Front-End | AD8232 |
| IMU | MPU6050 (I2C) |
| GPS | NEO-6M (UART/NMEA) |
| Backend Server | Python 3, Flask 3.0, Flask-SocketIO 5.3 |
| ML Framework | scikit-learn 1.3, XGBoost 2.0 |
| Data Processing | Pandas 2.0, NumPy 1.24 |
| Model Persistence | Joblib 1.3 |
| Frontend | HTML5, CSS3, JavaScript (ES6) |
| UI Framework | Bootstrap 5.3 |
| Charting | Chart.js 4.4 |
| Mapping | Leaflet.js 1.9 |
| Real-Time Comms | Socket.IO 4.7 |
| Visualization (Training) | Matplotlib 3.7, Seaborn 0.12 |

---

## 8. Use Case Diagram

```mermaid
graph TB
    subgraph "AI-Based Soldier Survival & Emergency Detection System"
        UC1((Transmit Sensor Data))
        UC2((Collect Training Data))
        UC3((Train ML Model))
        UC4((Classify Soldier State))
        UC5((Monitor Live Dashboard))
        UC6((Receive Alerts))
        UC7((Clear Alerts))
        UC8((View Historical Data))
        UC9((Track GPS Location))
    end
    
    Soldier[Soldier - Wears Device]
    Operator[Operator - Command Center]
    System[System - Automated]
    Developer[Developer - ML Engineer]
    
    Soldier --> UC1
    Soldier --> UC9
    Operator --> UC5
    Operator --> UC6
    Operator --> UC7
    Operator --> UC8
    System --> UC4
    Developer --> UC2
    Developer --> UC3
    UC1 --> UC4
    UC4 --> UC6
```

---

## 9. Class Diagram

```mermaid
classDiagram
    class ESP32SensorModule {
        -WiFiClient client
        -float bpm
        -float hrv_sdnn
        -float pitch
        -float roll
        -bool ecg_lead_off
        +setup()
        +loop()
        +processECG()
        +processIMU()
        +parseGPS()
        +sendData()
    }
    
    class FlaskServer {
        -Flask app
        -SocketIO socketio
        -MLModel model
        -AlertManager alerts
        +run()
        +receive_data()
        +broadcast_state()
    }
    
    class MLModel {
        -XGBClassifier classifier
        -StandardScaler scaler
        -LabelEncoder encoder
        -list feature_names
        +load_model()
        +predict(features)
        +get_probabilities(features)
    }
    
    class FeatureComputer {
        -deque bpm_buffer
        -deque accel_buffer
        +compute_windowed_features()
        +update_buffers()
        +get_feature_vector()
    }
    
    class ClassificationEngine {
        -MLModel model
        -FeatureComputer features
        -deque prediction_history
        +classify(sensor_data)
        +apply_smoothing()
        +get_confidence()
    }
    
    class AlertManager {
        -int consecutive_count
        -datetime last_alert_time
        -bool alert_active
        +check_alert_condition()
        +trigger_alert()
        +clear_alert()
    }
    
    class WebSocketHandler {
        -SocketIO socketio
        +emit_update()
        +emit_alert()
        +handle_clear_alert()
    }
    
    class ModelTrainer {
        -DataFrame data
        -dict models
        +clean_data()
        +train_models()
        +evaluate_models()
        +select_best_model()
        +save_artifacts()
    }
    
    ESP32SensorModule --> FlaskServer : HTTP POST
    FlaskServer --> FeatureComputer : uses
    FlaskServer --> ClassificationEngine : uses
    FlaskServer --> AlertManager : uses
    FlaskServer --> WebSocketHandler : uses
    ClassificationEngine --> MLModel : uses
    AlertManager --> WebSocketHandler : triggers
    ModelTrainer --> MLModel : creates
```

---

## 10. Data Flow Diagram

### Level 0 - Context Diagram

```mermaid
graph LR
    Soldier((Soldier with Device)) -->|Biometric & Location Data| System[AI-Based Soldier Survival System]
    System -->|Real-time Status & Alerts| Operator((Command Center Operator))
    Developer((ML Developer)) -->|Training Configuration| System
    System -->|Model Reports & Analytics| Developer
```

### Level 1 - Main Processes

```mermaid
graph TB
    subgraph "External Entities"
        ESP32[ESP32 Wearable Device]
        Dashboard[Web Dashboard]
        Developer[ML Developer]
    end
    
    subgraph "Data Stores"
        DS1[(soldier_data.csv)]
        DS2[(models/)]
        DS3[(Feature Buffers)]
    end
    
    subgraph "Processes"
        P1[1.0 Sensor Data Acquisition]
        P2[2.0 Feature Computation]
        P3[3.0 ML Classification]
        P4[4.0 Alert Management]
        P5[5.0 Dashboard Update]
        P6[6.0 Model Training]
    end
    
    ESP32 -->|Raw Sensor JSON| P1
    P1 -->|11 Raw Features| P2
    P1 -->|Labeled Data| DS1
    P2 -->|Buffer Update| DS3
    DS3 -->|Historical Data| P2
    P2 -->|19 Features| P3
    DS2 -->|Trained Model| P3
    P3 -->|Predicted State| P4
    P4 -->|Alert Event| P5
    P3 -->|Classification Result| P5
    P5 -->|WebSocket Push| Dashboard
    Developer -->|Training Command| P6
    DS1 -->|Training Data| P6
    P6 -->|Model Artifacts| DS2
```

---

## 11. Component Diagram

```mermaid
graph TB
    subgraph "Hardware Layer"
        ESP32[ESP32 Microcontroller]
        AD8232[AD8232 ECG Module]
        MPU6050[MPU6050 IMU]
        NEO6M[NEO-6M GPS]
        AD8232 -->|Analog| ESP32
        MPU6050 -->|I2C| ESP32
        NEO6M -->|UART| ESP32
    end
    
    subgraph "Application Layer"
        subgraph "Flask Server"
            Routes[HTTP Routes /data /api/*]
            FeatureEngine[Feature Computer]
            MLEngine[Classification Engine]
            AlertMgr[Alert Manager]
            SIOHandler[SocketIO Handler]
        end
        subgraph "ML Components"
            XGBoost[XGBoost Classifier]
            Scaler[Standard Scaler]
            Encoder[Label Encoder]
        end
    end
    
    subgraph "Presentation Layer"
        HTML[Dashboard HTML]
        ChartJS[Chart.js]
        Leaflet[Leaflet.js GPS Map]
        SIOClient[Socket.IO Client]
    end
    
    subgraph "Data Layer"
        CSV[(soldier_data.csv)]
        Models[(models/*.joblib)]
    end
    
    ESP32 -->|HTTP POST| Routes
    Routes --> FeatureEngine
    FeatureEngine --> MLEngine
    MLEngine --> XGBoost
    MLEngine --> Scaler
    MLEngine --> AlertMgr
    AlertMgr --> SIOHandler
    SIOHandler -->|WebSocket| SIOClient
    SIOClient --> HTML
    HTML --> ChartJS
    HTML --> Leaflet
    MLEngine --> Models
```

---

## 12. Sequence Diagram

### Real-Time Classification Flow

```mermaid
sequenceDiagram
    participant S as ESP32 Sensors
    participant F as Flask Server
    participant FC as Feature Computer
    participant ML as ML Classifier
    participant AM as Alert Manager
    participant WS as WebSocket
    participant D as Dashboard
    
    loop Every 1000ms (1 Hz)
        S->>S: Read ECG, IMU, GPS
        S->>F: HTTP POST /data (JSON)
        F->>FC: Update windowed buffers
        FC-->>F: 19-feature vector
        F->>ML: Classify(features)
        ML-->>F: State + Confidence
        F->>AM: Check alert condition
        alt State == Man Down (3 consecutive)
            AM->>WS: Emit alert event
        end
        F->>WS: Emit state update
        WS->>D: Push via WebSocket
        D->>D: Update UI, charts, map
    end
    
    opt Operator clears alert
        D->>WS: clear_alert event
        WS->>AM: Clear alert
        AM->>WS: Alert cleared
        WS->>D: Update UI
    end
```

### Model Training Flow

```mermaid
sequenceDiagram
    participant D as Developer
    participant DC as data_collection.py
    participant E as ESP32
    participant MT as model_training.py
    participant FS as File System
    
    Note over D,E: Phase 1: Data Collection
    D->>DC: Set CURRENT_LABEL, Run
    loop For each label
        E->>DC: POST sensor data
        DC->>FS: Append to soldier_data.csv
    end
    
    Note over D,FS: Phase 2: Model Training
    D->>MT: Run model_training.py
    MT->>FS: Load soldier_data.csv
    MT->>MT: Clean data, Extract features
    MT->>MT: Train 4 models (RF, XGB, SVM, MLP)
    MT->>MT: 5-Fold Cross-Validation
    MT->>MT: Select best model
    MT->>FS: Save artifacts to models/
    MT-->>D: Training complete report
```

---

## 13. Deployment Diagram

```mermaid
graph TB
    subgraph "Field - Soldier Wearable"
        subgraph "ESP32 DevKit"
            CPU[CPU 240MHz Dual Core]
            WiFi[WiFi 802.11 b/g/n]
        end
        subgraph "Sensors"
            ECG[AD8232 ECG]
            IMU[MPU6050 IMU]
            GPS[NEO-6M GPS]
        end
        ECG --> CPU
        IMU --> CPU
        GPS --> CPU
    end
    
    subgraph "Command Center - Base Station"
        subgraph "Server Machine"
            OS[Python 3.10+ / Port 5000]
            subgraph "Flask Application"
                FlaskApp[Flask Server + SocketIO]
                MLRuntime[XGBoost + sklearn]
            end
            ModelFiles[(models/*.joblib)]
            FlaskApp --> MLRuntime
            MLRuntime --> ModelFiles
        end
        Browser[Operator Browser<br/>Chrome/Firefox/Edge]
    end
    
    WiFi -->|HTTP POST 1Hz| FlaskApp
    FlaskApp -->|WebSocket Real-time| Browser
```

---

## 14. ER Diagram

```mermaid
erDiagram
    SOLDIER ||--o{ SESSION : participates_in
    SESSION ||--o{ SENSOR_READING : contains
    SENSOR_READING ||--|| ECG_DATA : has
    SENSOR_READING ||--|| IMU_DATA : has
    SENSOR_READING ||--|| GPS_DATA : has
    SESSION }o--|| ACTIVITY_LABEL : labeled_as
    ML_MODEL ||--o{ PREDICTION : generates
    SENSOR_READING ||--o{ PREDICTION : input_for
    PREDICTION ||--o| ALERT : may_trigger
    
    SOLDIER {
        string soldier_id PK
        string name
        string unit
        string device_id FK
    }
    
    SESSION {
        string session_id PK
        string soldier_id FK
        datetime start_time
        datetime end_time
        string label
    }
    
    SENSOR_READING {
        int reading_id PK
        string session_id FK
        float timestamp
        bool ecg_lead_off
    }
    
    ECG_DATA {
        int reading_id PK
        float bpm
        float hrv_sdnn
        float hrv_rmssd
    }
    
    IMU_DATA {
        int reading_id PK
        float dynamic_accel
        float impact
        float pitch
        float roll
        float movement_var
    }
    
    GPS_DATA {
        int reading_id PK
        float latitude
        float longitude
        float speed
        float altitude
        int satellites
    }
    
    ML_MODEL {
        string model_id PK
        string model_type
        float accuracy
        float f1_score
        datetime trained_at
    }
    
    PREDICTION {
        int prediction_id PK
        int reading_id FK
        string model_id FK
        string predicted_state
        float confidence
    }
    
    ALERT {
        int alert_id PK
        int prediction_id FK
        string alert_type
        datetime triggered_at
        datetime cleared_at
        float gps_lat
        float gps_lon
    }
```

---

## 15. Schema Design

### Training Dataset Schema (soldier_data.csv)

```sql
CREATE TABLE sensor_readings (
    timestamp         DECIMAL(18,6) PRIMARY KEY,
    session_id        VARCHAR(50) NOT NULL,
    subject_id        VARCHAR(20) NOT NULL,
    
    -- ECG Features
    bpm               DECIMAL(5,2),
    hrv_sdnn          DECIMAL(6,2),
    hrv_rmssd         DECIMAL(6,2),
    ecg_lead_off      BOOLEAN DEFAULT FALSE,
    
    -- IMU Features
    smv               DECIMAL(6,4),
    dynamic_accel     DECIMAL(6,4),
    impact            DECIMAL(6,4),
    pitch             DECIMAL(6,2),
    roll              DECIMAL(6,2),
    gx                DECIMAL(8,2),
    gy                DECIMAL(8,2),
    gz                DECIMAL(8,2),
    movement_var      DECIMAL(10,8),
    
    -- Windowed Features
    bpm_mean_10s             DECIMAL(5,2),
    bpm_std_10s              DECIMAL(5,2),
    dynamic_accel_mean_5s    DECIMAL(6,4),
    dynamic_accel_max_5s     DECIMAL(6,4),
    impact_max_5s            DECIMAL(6,4),
    pitch_mean_5s            DECIMAL(6,2),
    movement_var_mean_5s     DECIMAL(10,8),
    gyro_magnitude_mean_5s   DECIMAL(8,2),
    
    -- GPS
    gps_lat           DECIMAL(10,7),
    gps_lon           DECIMAL(11,7),
    gps_speed         DECIMAL(6,2),
    gps_alt           DECIMAL(7,2),
    gps_satellites    TINYINT,
    gps_fix           BOOLEAN,
    
    -- Label
    label             VARCHAR(20) NOT NULL
);
```

---

## 16. Data Exchange Contract

### Frequency of Data Exchanges

| Exchange | Direction | Frequency | Trigger |
|----------|-----------|-----------|---------|
| Sensor Data | ESP32 → Server | 1 Hz (1000ms) | Timer-based |
| Dashboard Update | Server → Dashboard | 1 Hz (1000ms) | On sensor receipt |
| Alert Notification | Server → Dashboard | Event-driven | On alert trigger |
| Alert Clear | Dashboard → Server | Event-driven | User action |
| Status Request | Dashboard → Server | On-demand | API call |

### Data Sets

**Sensor Data Payload (ESP32 → Server):**
```json
{
    "bpm": 72.5, "hrv_sdnn": 45.2, "hrv_rmssd": 38.1,
    "smv": 1.02, "dynamic_accel": 0.02, "impact": 0.15,
    "pitch": 5.3, "roll": -2.1,
    "gx": 1.2, "gy": -0.5, "gz": 0.8,
    "movement_var": 0.0003,
    "gps_lat": 28.6139, "gps_lon": 77.2090,
    "gps_speed": 0.0, "gps_alt": 216.0,
    "gps_satellites": 7, "gps_fix": 1,
    "ecg_lead_off": 0
}
```

**Dashboard Update Payload (Server → Dashboard via WebSocket):**
```json
{
    "status": "normal",
    "confidence": {"normal": 0.95, "high_exertion": 0.03, "man_down": 0.02},
    "vitals": {"bpm": 72.5, "hrv_sdnn": 45.2, "hrv_rmssd": 38.1},
    "motion": {"dynamic_accel": 0.02, "impact": 0.15, "pitch": 5.3, "roll": -2.1},
    "gps": {"lat": 28.6139, "lon": 77.2090, "speed": 0.0},
    "alert_active": false,
    "timestamp": 1772701958.123
}
```

### Mode of Exchanges

| # | Exchange | Mode | Format | Protocol |
|---|----------|------|--------|----------|
| 1 | Sensor Data | HTTP POST | JSON | REST API |
| 2 | Dashboard Update | WebSocket Push | JSON | Socket.IO |
| 3 | Alert Notification | WebSocket Push | JSON | Socket.IO |
| 4 | Alert Clear | WebSocket Event | JSON | Socket.IO |
| 5 | Historical Data | HTTP GET | JSON | REST API |
| 6 | Training Data | File Write | CSV | File System |
| 7 | Model Artifacts | File Read/Write | Joblib | File System |

### Data Exchange Diagram

```mermaid
graph TB
    subgraph "HTTP REST API"
        ESP32[ESP32] -->|POST /data JSON 1Hz| Flask[Flask Server]
        Browser[Dashboard] -->|GET /api/*| Flask
    end
    
    subgraph "WebSocket Socket.IO"
        Flask -->|state_update 1Hz| SIO[Socket.IO]
        SIO --> Dashboard[Dashboard]
        Flask -->|alert event| SIO
        Dashboard -->|clear_alert| SIO
    end
    
    subgraph "File Storage"
        Flask -->|Append| CSV[(soldier_data.csv)]
        Trainer[model_training.py] -->|Write| Models[(models/)]
        Flask -->|Load| Models
    end
```

---
