# Architecture Diagrams

**Project:** AI-Based Soldier Survival & Emergency Detection System

---

## 1. Event-Driven Architecture Diagram

```mermaid
graph TB
    subgraph Producers
        ESP32[ESP32 Wearable Sensors]
    end
    
    subgraph Broker
        subgraph FlaskServer[Flask Server]
            Queue[Event Queue]
            Processor[ML Classifier]
            Router[Event Router]
            Store[Event Store]
        end
    end
    
    subgraph Consumers
        Dashboard[Web Dashboard]
        AlertHandler[Alert Handler]
    end
    
    ESP32 -->|HTTP POST| Queue
    Queue --> Processor
    Processor --> Router
    Processor --> Store
    Router -->|WebSocket| Dashboard
    Router -->|Alert| AlertHandler
    AlertHandler --> Dashboard
```

---

## 2. On-Device Processing Pipeline

```mermaid
graph LR
    subgraph ESP32
        ECG[Raw ECG] --> AR[Artifact Rejection]
        AR --> BT[Baseline Tracking]
        BT --> TD[Threshold Detection]
        TD --> RP[R-Peak]
        RP --> BPM[BPM Filter]
        BPM --> HRV[HRV Metrics]
        
        IMU[Raw IMU] --> GC[G-Conversion]
        GC --> SMV[SMV]
        SMV --> DA[Dynamic Accel]
        DA --> Impact[Impact]
        Impact --> PR[Pitch Roll]
        PR --> MV[Movement Var]
        
        GPS[Raw NMEA] --> Parse[Parse]
        Parse --> Coords[Coordinates]
    end
    
    HRV --> JSON[JSON Payload]
    MV --> JSON
    Coords --> JSON
    JSON -->|HTTP 1Hz| Server[Flask Server]
```

---

## 3. ML Pipeline Architecture

```mermaid
graph TB
    CSV[(soldier_data.csv)] --> Clean[Data Cleaning]
    Clean --> Extract[Feature Extraction]
    Extract --> Split[Train Test Split]
    Split --> Scale[StandardScaler]
    
    Scale --> RF[Random Forest]
    Scale --> XGB[XGBoost]
    Scale --> SVM[SVM]
    Scale --> MLP[Neural Network]
    
    RF --> CV[Cross Validation]
    XGB --> CV
    SVM --> CV
    MLP --> CV
    
    CV --> Select[Select Best Model]
    Select --> Save[(models/)]
```

---

## 4. Real-Time Inference Flow

```mermaid
graph TB
    ESP32[ESP32 POST] --> Handler[Flask Handler]
    Handler --> Buffer[Update Buffers]
    Buffer --> Compute[Compute Features]
    Compute --> Vector[Feature Vector]
    Vector --> Scale[Scaler Transform]
    Scale --> Predict[XGBoost Predict]
    Predict --> Smooth[Majority Vote]
    Smooth --> Alert{Man Down x3?}
    Alert -->|Yes| Critical[Critical Alert]
    Alert -->|No| WS[WebSocket Emit]
    Critical --> WS
    WS --> Dashboard[Dashboard]
```

---

## 5. Use Case Diagram

```mermaid
graph TB
    subgraph System
        UC1((Transmit Data))
        UC2((Collect Training Data))
        UC3((Train Model))
        UC4((Classify State))
        UC5((Monitor Dashboard))
        UC6((Receive Alerts))
        UC7((Clear Alerts))
        UC8((View History))
        UC9((Track Location))
    end
    
    Soldier[Soldier]
    Operator[Operator]
    SystemActor[System]
    Developer[Developer]
    
    Soldier --> UC1
    Soldier --> UC9
    Operator --> UC5
    Operator --> UC6
    Operator --> UC7
    Operator --> UC8
    SystemActor --> UC4
    Developer --> UC2
    Developer --> UC3
    UC1 --> UC4
    UC4 --> UC6
```

---

## 6. Class Diagram

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
        +predict()
        +get_probabilities()
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
        +classify()
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

## 7. Data Flow Diagram - Level 0

```mermaid
graph LR
    Soldier((Soldier)) -->|Biometric Data| System[Soldier Survival System]
    System -->|Status and Alerts| Operator((Operator))
    Developer((Developer)) -->|Training Config| System
    System -->|Reports| Developer
```

---

## 8. Data Flow Diagram - Level 1

```mermaid
graph TB
    subgraph External
        ESP32[ESP32 Device]
        Dashboard[Web Dashboard]
        Developer[Developer]
    end
    
    subgraph DataStores
        DS1[(soldier_data.csv)]
        DS2[(models/)]
        DS3[(Feature Buffers)]
    end
    
    subgraph Processes
        P1[1.0 Data Acquisition]
        P2[2.0 Feature Computation]
        P3[3.0 ML Classification]
        P4[4.0 Alert Management]
        P5[5.0 Dashboard Update]
        P6[6.0 Model Training]
    end
    
    ESP32 -->|Raw JSON| P1
    P1 -->|Raw Features| P2
    P1 -->|Labeled Data| DS1
    P2 -->|Buffer Update| DS3
    DS3 -->|History| P2
    P2 -->|19 Features| P3
    DS2 -->|Model| P3
    P3 -->|State| P4
    P4 -->|Alert| P5
    P3 -->|Result| P5
    P5 -->|WebSocket| Dashboard
    Developer -->|Command| P6
    DS1 -->|Training Data| P6
    P6 -->|Artifacts| DS2
```

---

## 9. Component Diagram

```mermaid
graph TB
    subgraph Hardware
        ESP32[ESP32]
        AD8232[AD8232 ECG]
        MPU6050[MPU6050 IMU]
        NEO6M[NEO-6M GPS]
        AD8232 -->|Analog| ESP32
        MPU6050 -->|I2C| ESP32
        NEO6M -->|UART| ESP32
    end
    
    subgraph Application
        subgraph Flask
            Routes[HTTP Routes]
            FeatureEngine[Feature Computer]
            MLEngine[Classification Engine]
            AlertMgr[Alert Manager]
            SIOHandler[SocketIO Handler]
        end
        subgraph ML
            XGBoost[XGBoost]
            Scaler[Scaler]
            Encoder[Encoder]
        end
    end
    
    subgraph Presentation
        HTML[Dashboard]
        ChartJS[Charts]
        Leaflet[GPS Map]
        SIOClient[Socket.IO Client]
    end
    
    subgraph Data
        CSV[(soldier_data.csv)]
        Models[(models/)]
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

## 10. Sequence Diagram - Real-Time Classification

```mermaid
sequenceDiagram
    participant S as ESP32
    participant F as Flask Server
    participant FC as Feature Computer
    participant ML as ML Classifier
    participant AM as Alert Manager
    participant WS as WebSocket
    participant D as Dashboard
    
    loop Every 1000ms
        S->>S: Read Sensors
        S->>F: HTTP POST /data
        F->>FC: Update Buffers
        FC-->>F: Feature Vector
        F->>ML: Classify
        ML-->>F: State + Confidence
        F->>AM: Check Alert
        alt Man Down x3
            AM->>WS: Emit Alert
        end
        F->>WS: Emit State
        WS->>D: Push Update
        D->>D: Update UI
    end
    
    opt Clear Alert
        D->>WS: clear_alert
        WS->>AM: Clear
        AM->>WS: Cleared
        WS->>D: Update UI
    end
```

---

## 11. Sequence Diagram - Model Training

```mermaid
sequenceDiagram
    participant D as Developer
    participant DC as data_collection.py
    participant E as ESP32
    participant MT as model_training.py
    participant FS as File System
    
    Note over D,E: Data Collection Phase
    D->>DC: Set Label and Run
    loop For Each Label
        E->>DC: POST Sensor Data
        DC->>FS: Append to CSV
    end
    
    Note over D,FS: Training Phase
    D->>MT: Run Training
    MT->>FS: Load CSV
    MT->>MT: Clean Data
    MT->>MT: Train 4 Models
    MT->>MT: Cross Validation
    MT->>MT: Select Best
    MT->>FS: Save Artifacts
    MT-->>D: Training Report
```

---

## 12. Deployment Diagram

```mermaid
graph TB
    subgraph Field
        subgraph ESP32DevKit[ESP32 DevKit]
            CPU[Dual Core 240MHz]
            WiFi[WiFi]
        end
        subgraph Sensors
            ECG[AD8232]
            IMU[MPU6050]
            GPS[NEO-6M]
        end
        ECG --> CPU
        IMU --> CPU
        GPS --> CPU
    end
    
    subgraph CommandCenter[Command Center]
        subgraph Server
            OS[Python 3.10 Port 5000]
            subgraph FlaskApp[Flask Application]
                App[Flask + SocketIO]
                Runtime[XGBoost + sklearn]
            end
            ModelFiles[(models/)]
            App --> Runtime
            Runtime --> ModelFiles
        end
        Browser[Operator Browser]
    end
    
    WiFi -->|HTTP POST 1Hz| App
    App -->|WebSocket| Browser
```

---

## 13. ER Diagram

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

## 14. Data Exchange Diagram

```mermaid
graph TB
    subgraph HTTP
        ESP32[ESP32] -->|POST /data 1Hz| Flask[Flask Server]
        Browser[Dashboard] -->|GET /api/*| Flask
    end
    
    subgraph WebSocket
        Flask -->|state_update 1Hz| SIO[Socket.IO]
        SIO --> Dashboard[Dashboard]
        Flask -->|alert| SIO
        Dashboard -->|clear_alert| SIO
    end
    
    subgraph Storage
        Flask -->|Append| CSV[(soldier_data.csv)]
        Trainer[model_training.py] -->|Write| Models[(models/)]
        Flask -->|Load| Models
    end
```

---
