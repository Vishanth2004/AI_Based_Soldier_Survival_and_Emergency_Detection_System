/*
 * AI-Based Soldier Survival & Emergency Detection System
 * ESP32 Sensor Hub — AD8232 + MPU6050 + NEO-6M GPS
 * 
 * Wiring:
 *   AD8232:  OUTPUT -> GPIO34, LO+ -> GPIO32, LO- -> GPIO33, 3.3V, GND
 *   MPU6050: SDA -> GPIO21, SCL -> GPIO22, 3.3V, GND
 *   NEO-6M:  TX -> GPIO16 (ESP RX2), RX -> GPIO17 (ESP TX2), 3.3V/5V, GND
 *   
 * Sends JSON data to Python server over WiFi every ~1000ms (~1 Hz for features)
 * Raw ECG sampled at ~166 Hz internally for R-peak detection
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>
#include <HardwareSerial.h>

// ==================== CONFIGURATION ====================
const char* WIFI_SSID     = "Airte";
const char* WIFI_PASSWORD  = "dharun13";
const char* SERVER_URL     = "http://10.38.15.156:5001/data";

// ==================== PIN DEFINITIONS ====================
#define ECG_PIN    34
#define LO_PLUS    32
#define LO_MINUS   33
#define MPU_ADDR   0x68

// ==================== GPS SERIAL ====================
HardwareSerial gpsSerial(2);  // UART2: RX=16, TX=17

// ==================== ECG / BPM VARIABLES ====================

// Debug / raw ECG
int lastRawECG = 0;

// Baseline-tracking BPM detection (works on raw ADC directly)
float ecgBase = 2048.0;       // Slow-moving baseline of raw ECG
float ecgPeakDev = 100.0;     // Tracks how high R-peaks go above baseline
float ecgThreshold = 2048;    // Dynamic threshold for beat detection
bool ecgAboveThreshold = false;

unsigned long lastRPeakTime = 0;
unsigned long rrIntervals[20];
int rrIndex = 0;
int rrCount = 0;

float currentBPM = 0;
float smoothBPM = 0;       // Median-filtered BPM (stable output)
float recentBPMs[7];          // Last 7 BPM readings for median filter
int bpmBufIdx = 0;
int bpmBufCount = 0;
float currentHRV_SDNN = 0;
float currentHRV_RMSSD = 0;
bool ecgLeadOff = true;
int artifactCount = 0;        // Debug: how many samples rejected as artifacts

// Lead-off raw pin states (debug)
int loPlusState = 1;
int loMinusState = 1;

unsigned long lastBeatTime = 0;
int beatCount = 0;

// HTTP backoff (prevents blocking ECG sampling when server is unreachable)
int httpFailCount = 0;
unsigned long httpBackoffUntil = 0;
const int HTTP_MAX_FAILS = 3;
const unsigned long HTTP_BACKOFF_MS = 10000;  // 10 sec backoff after repeated failures
int beatDetections = 0;  // Debug: total threshold crossings detected

// ==================== MPU6050 VARIABLES ====================
float ax_g, ay_g, az_g;
float gx_dps, gy_dps, gz_dps;
float smv = 0;
float dynamicAccel = 0;
float prevSMV = 0;
float impact = 0;
float pitch = 0;
float roll = 0;

// Movement variance (over window)
#define ACCEL_WINDOW 100
float accelWindow[ACCEL_WINDOW];
int accelWinIdx = 0;
int accelWinCount = 0;
float movementVariance = 0;

// ==================== GPS VARIABLES ====================
float gpsLat = 0.0;
float gpsLon = 0.0;
float gpsSpeed = 0.0;  // km/h
float gpsAlt = 0.0;
int gpsSatellites = 0;
bool gpsFix = false;
char gpsBuffer[256];
int gpsBufferIdx = 0;

// ==================== TIMING ====================
unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 1000;  // 1 Hz data transmission (enough for ML features)

unsigned long lastECGSample = 0;
const unsigned long ECG_INTERVAL = 6;  // ~166 Hz ECG sampling

unsigned long lastDebugPrint = 0;
const unsigned long DEBUG_INTERVAL = 1000;  // 1 Hz debug output (Serial Monitor)

// ==================== SETUP ====================
void setup() {
    Serial.begin(115200);
    
    // ECG pins
    // AD8232 LO pins can float if not wired well; use pulldown to avoid false lead-off.
    pinMode(LO_PLUS, INPUT_PULLDOWN);
    pinMode(LO_MINUS, INPUT_PULLDOWN);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    
    // I2C for MPU6050
    Wire.begin(21, 22);
    Wire.setClock(400000);  // 400 kHz I2C
    
    // Wake MPU6050
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);
    Wire.write(0);
    Wire.endTransmission(true);
    
    // Set accelerometer to ±4g (for better impact detection)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C);
    Wire.write(0x08);  // ±4g
    Wire.endTransmission(true);
    
    // Set gyroscope to ±500°/s
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1B);
    Wire.write(0x08);  // ±500°/s
    Wire.endTransmission(true);
    
    // GPS Serial
    gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
    
    // WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());
    
    // Initialize arrays
    memset(rrIntervals, 0, sizeof(rrIntervals));
    memset(accelWindow, 0, sizeof(accelWindow));
    
    lastRPeakTime = millis();
    lastBeatTime = millis();
}

// ==================== ECG PROCESSING ====================
void sampleECG() {
    loPlusState = digitalRead(LO_PLUS);
    loMinusState = digitalRead(LO_MINUS);
    ecgLeadOff = (loPlusState == 1 || loMinusState == 1);

    int rawECG = analogRead(ECG_PIN);
    lastRawECG = rawECG;
    
    if (ecgLeadOff) return;

    // === MOTION ARTIFACT REJECTION ===
    // ADC saturation (0 or 4095) = electrode physically moved = guaranteed artifact
    if (rawECG <= 5 || rawECG >= 4090) {
        artifactCount++;
        return;  // Skip this sample entirely
    }
    // Huge jump from baseline = motion artifact, not a heartbeat
    // Normal R-peak deviation is ~500-800; anything >1200 is artifact
    float jumpFromBase = fabs((float)rawECG - ecgBase);
    if (jumpFromBase > 1200.0f) {
        artifactCount++;
        return;  // Skip — don't let it corrupt baseline or peak tracking
    }

    // === BASELINE TRACKING ===
    // Very slow moving average — follows baseline drift, NOT the QRS peaks
    // At 166 Hz, alpha=0.002 → time constant ~3 seconds
    ecgBase = 0.998f * ecgBase + 0.002f * (float)rawECG;

    // How far is this sample above the baseline?
    float deviation = (float)rawECG - ecgBase;

    // === PEAK DEVIATION TRACKING ===
    // Track the typical R-peak height above baseline
    // Soft attack: blends new peak with old — one noise spike can't corrupt it
    // e.g. if Peak=500 and spike=2000: 0.7*500 + 0.3*2000 = 950 (not 2000)
    //      but real R-peaks at ~600 quickly converge: 0.7*500 + 0.3*600 = 530
    if (deviation > ecgPeakDev) {
        ecgPeakDev = 0.7f * ecgPeakDev + 0.3f * deviation;
    }
    // Faster decay: at 166 Hz, 0.999^166 ≈ 0.85 → ~15% decay per second
    // Recovers from a bad spike in ~3-4 seconds instead of ~15+
    ecgPeakDev *= 0.999f;
    // Floor: don't let it collapse to zero
    if (ecgPeakDev < 30.0f) ecgPeakDev = 30.0f;

    // === THRESHOLD ===
    // Set at 55% of peak deviation above baseline (higher = fewer false positives)
    ecgThreshold = ecgBase + ecgPeakDev * 0.55f;

    // === BEAT DETECTION ===
    unsigned long now = millis();
    bool refractoryOk = (now - lastRPeakTime) > 400;  // max ~150 BPM (was 300/200)

    if (!ecgAboveThreshold && rawECG > ecgThreshold && refractoryOk) {
        // Rising edge — heartbeat detected
        ecgAboveThreshold = true;
        beatDetections++;  // Debug counter

        unsigned long rrInterval = now - lastRPeakTime;

        if (lastRPeakTime > 0 && rrInterval > 400 && rrInterval < 2000) {
            // RR consistency check: reject if >50% different from recent median RR
            // This prevents noise-triggered false beats from corrupting BPM
            bool rrConsistent = true;
            if (rrCount >= 3) {
                // Get median of recent RR intervals
                float rrSorted[20];
                int n = rrCount;
                for (int i = 0; i < n; i++) rrSorted[i] = (float)rrIntervals[i];
                for (int i = 1; i < n; i++) {
                    float key = rrSorted[i];
                    int j = i - 1;
                    while (j >= 0 && rrSorted[j] > key) { rrSorted[j+1] = rrSorted[j]; j--; }
                    rrSorted[j+1] = key;
                }
                float medianRR = rrSorted[n / 2];
                float rrDiffRatio = fabs((float)rrInterval - medianRR) / medianRR;
                if (rrDiffRatio > 0.50f) rrConsistent = false;  // >50% deviation = reject
            }

            if (rrConsistent) {
                rrIntervals[rrIndex] = rrInterval;
                rrIndex = (rrIndex + 1) % 20;
                if (rrCount < 20) rrCount++;
                currentBPM = 60000.0 / rrInterval;

                // Store in median buffer for smoothing
                recentBPMs[bpmBufIdx] = currentBPM;
                bpmBufIdx = (bpmBufIdx + 1) % 7;
                if (bpmBufCount < 7) bpmBufCount++;

                // Compute median BPM (sorts a small copy)
                float sorted[7];
                int nn = bpmBufCount;
                for (int i = 0; i < nn; i++) sorted[i] = recentBPMs[i];
                // Simple insertion sort (only 7 elements)
                for (int i = 1; i < nn; i++) {
                    float key = sorted[i];
                    int j = i - 1;
                    while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j--; }
                    sorted[j+1] = key;
                }
                smoothBPM = sorted[nn / 2];  // Median value
            }  // end rrConsistent
        }  // end rrInterval range check

        lastRPeakTime = now;
    }

    // Re-arm when signal drops below threshold
    if (ecgAboveThreshold && rawECG < ecgThreshold) {
        ecgAboveThreshold = false;
    }
}

void computeHRV() {
    if (rrCount < 5) {
        currentHRV_SDNN = 0;
        currentHRV_RMSSD = 0;
        return;
    }
    
    // SDNN: Standard deviation of R-R intervals
    float meanRR = 0;
    for (int i = 0; i < rrCount; i++) {
        meanRR += rrIntervals[i];
    }
    meanRR /= rrCount;
    
    float sumSqDiff = 0;
    for (int i = 0; i < rrCount; i++) {
        float diff = rrIntervals[i] - meanRR;
        sumSqDiff += diff * diff;
    }
    currentHRV_SDNN = sqrt(sumSqDiff / rrCount);
    
    // RMSSD: Root mean square of successive differences
    float sumSqSuccDiff = 0;
    int pairs = 0;
    for (int i = 1; i < rrCount; i++) {
        float diff = (float)rrIntervals[i] - (float)rrIntervals[i - 1];
        sumSqSuccDiff += diff * diff;
        pairs++;
    }
    if (pairs > 0) {
        currentHRV_RMSSD = sqrt(sumSqSuccDiff / pairs);
    }
}

// ==================== MPU6050 PROCESSING ====================
void readMPU6050() {
    // Read accelerometer (registers 0x3B-0x40)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14, true);  // Read accel + temp + gyro
    
    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    
    int16_t temp_raw = Wire.read() << 8 | Wire.read();  // Skip temperature
    
    int16_t gx = Wire.read() << 8 | Wire.read();
    int16_t gy = Wire.read() << 8 | Wire.read();
    int16_t gz = Wire.read() << 8 | Wire.read();
    
    // Convert to physical units (±4g range -> 8192 LSB/g)
    ax_g = ax / 8192.0;
    ay_g = ay / 8192.0;
    az_g = az / 8192.0;
    
    // Convert gyro (±500°/s -> 65.5 LSB/°/s)
    gx_dps = gx / 65.5;
    gy_dps = gy / 65.5;
    gz_dps = gz / 65.5;
    
    // Signal Magnitude Vector
    prevSMV = smv;
    smv = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
    
    // Dynamic acceleration (remove gravity component)
    dynamicAccel = fabs(smv - 1.0);
    
    // Impact / Jerk (change in acceleration)
    impact = fabs(smv - prevSMV) / (SEND_INTERVAL / 1000.0);
    
    // Pitch and Roll (body orientation)
    pitch = atan2(ay_g, sqrt(ax_g * ax_g + az_g * az_g)) * 180.0 / M_PI;
    roll  = atan2(ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * 180.0 / M_PI;
    
    // Movement variance window
    accelWindow[accelWinIdx] = dynamicAccel;
    accelWinIdx = (accelWinIdx + 1) % ACCEL_WINDOW;
    if (accelWinCount < ACCEL_WINDOW) accelWinCount++;
    
    // Compute variance
    float mean = 0;
    for (int i = 0; i < accelWinCount; i++) mean += accelWindow[i];
    mean /= accelWinCount;
    
    float varSum = 0;
    for (int i = 0; i < accelWinCount; i++) {
        float d = accelWindow[i] - mean;
        varSum += d * d;
    }
    movementVariance = varSum / accelWinCount;
}

// ==================== GPS PROCESSING ====================
void readGPS() {
    while (gpsSerial.available()) {
        char c = gpsSerial.read();
        
        if (c == '\n') {
            gpsBuffer[gpsBufferIdx] = '\0';
            parseNMEA(gpsBuffer);
            gpsBufferIdx = 0;
        } else if (gpsBufferIdx < 255) {
            gpsBuffer[gpsBufferIdx++] = c;
        }
    }
}

void parseNMEA(char* sentence) {
    // Parse $GPGGA for position, altitude, satellites
    if (strncmp(sentence, "$GPGGA", 6) == 0 || strncmp(sentence, "$GNGGA", 6) == 0) {
        char* token = strtok(sentence, ",");
        int fieldIdx = 0;
        char latStr[16] = "", lonStr[16] = "";
        char latDir = 'N', lonDir = 'E';
        
        while (token != NULL) {
            switch (fieldIdx) {
                case 2: strncpy(latStr, token, 15); break;
                case 3: latDir = token[0]; break;
                case 4: strncpy(lonStr, token, 15); break;
                case 5: lonDir = token[0]; break;
                case 6: gpsFix = (atoi(token) > 0); break;
                case 7: gpsSatellites = atoi(token); break;
                case 9: gpsAlt = atof(token); break;
            }
            token = strtok(NULL, ",");
            fieldIdx++;
        }
        
        if (gpsFix && strlen(latStr) > 0 && strlen(lonStr) > 0) {
            gpsLat = nmeaToDecimal(latStr, latDir);
            gpsLon = nmeaToDecimal(lonStr, lonDir);
        }
    }
    
    // Parse $GPVTG for speed
    if (strncmp(sentence, "$GPVTG", 6) == 0 || strncmp(sentence, "$GNVTG", 6) == 0) {
        char* token = strtok(sentence, ",");
        int fieldIdx = 0;
        
        while (token != NULL) {
            if (fieldIdx == 7) {  // Speed in km/h
                gpsSpeed = atof(token);
            }
            token = strtok(NULL, ",");
            fieldIdx++;
        }
    }
}

float nmeaToDecimal(char* nmeaCoord, char dir) {
    float raw = atof(nmeaCoord);
    int degrees = (int)(raw / 100);
    float minutes = raw - (degrees * 100);
    float decimal = degrees + (minutes / 60.0);
    if (dir == 'S' || dir == 'W') decimal = -decimal;
    return decimal;
}

// ==================== SEND DATA ====================
void sendData() {
    if (WiFi.status() != WL_CONNECTED) return;
    
    computeHRV();
    
    // Skip sending if in backoff (server unreachable)
    // This is CRITICAL — HTTP timeouts block the main loop and prevent ECG sampling
    unsigned long now = millis();
    if (now < httpBackoffUntil) return;
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(200);  // Reduced from 500ms — less blocking of ECG sampling
    
    // Build JSON manually (to avoid ArduinoJson dependency)
    String json = "{";
    json += "\"bpm\":" + String(smoothBPM, 1) + ",";
    json += "\"hrv_sdnn\":" + String(currentHRV_SDNN, 2) + ",";
    json += "\"hrv_rmssd\":" + String(currentHRV_RMSSD, 2) + ",";
    json += "\"smv\":" + String(smv, 4) + ",";
    json += "\"dynamic_accel\":" + String(dynamicAccel, 4) + ",";
    json += "\"impact\":" + String(impact, 4) + ",";
    json += "\"pitch\":" + String(pitch, 2) + ",";
    json += "\"roll\":" + String(roll, 2) + ",";
    json += "\"gx\":" + String(gx_dps, 2) + ",";
    json += "\"gy\":" + String(gy_dps, 2) + ",";
    json += "\"gz\":" + String(gz_dps, 2) + ",";
    json += "\"movement_var\":" + String(movementVariance, 6) + ",";
    json += "\"ecg_lead_off\":" + String(ecgLeadOff ? 1 : 0) + ",";
    json += "\"gps_lat\":" + String(gpsLat, 6) + ",";
    json += "\"gps_lon\":" + String(gpsLon, 6) + ",";
    json += "\"gps_speed\":" + String(gpsSpeed, 2) + ",";
    json += "\"gps_alt\":" + String(gpsAlt, 1) + ",";
    json += "\"gps_satellites\":" + String(gpsSatellites) + ",";
    json += "\"gps_fix\":" + String(gpsFix ? 1 : 0);
    json += "}";
    
    int httpCode = http.POST(json);
    
    if (httpCode > 0) {
        httpFailCount = 0;  // Reset on success
    } else {
        httpFailCount++;
        if (httpFailCount >= HTTP_MAX_FAILS) {
            httpBackoffUntil = millis() + HTTP_BACKOFF_MS;
            Serial.printf("HTTP: %d consecutive fails — backing off %d sec\n",
                          httpFailCount, (int)(HTTP_BACKOFF_MS / 1000));
            httpFailCount = 0;
        } else {
            Serial.println("HTTP Error: " + http.errorToString(httpCode));
        }
    }
    
    http.end();
}

// ==================== MAIN LOOP ====================
void loop() {
    unsigned long now = millis();
    
    // Sample ECG at high rate (~166 Hz)
    if (now - lastECGSample >= ECG_INTERVAL) {
        sampleECG();
        lastECGSample = now;
    }
    
    // Read GPS continuously
    readGPS();
    
    // Send combined data at 20 Hz (needed for ML — captures fast motion dynamics)
    if (now - lastSendTime >= SEND_INTERVAL) {
        readMPU6050();
        sendData();
        lastSendTime = now;
    }
    
    // Debug output at 1 Hz (readable in Serial Monitor)
    if (now - lastDebugPrint >= DEBUG_INTERVAL) {
        Serial.printf(
            "BPM:%.0f Beats:%d Art:%d LO:%d Raw:%d Base:%.0f Peak:%.0f Thr:%.0f SMV:%.2f DynA:%.3f P:%.1f R:%.1f GPS:%s\n",
            smoothBPM,
            beatDetections,
            artifactCount,
            ecgLeadOff ? 1 : 0,
            lastRawECG,
            ecgBase,
            ecgPeakDev,
            ecgThreshold,
            smv,
            dynamicAccel,
            pitch,
            roll,
            gpsFix ? "FIX" : "NO"
        );
        lastDebugPrint = now;
    }
}
