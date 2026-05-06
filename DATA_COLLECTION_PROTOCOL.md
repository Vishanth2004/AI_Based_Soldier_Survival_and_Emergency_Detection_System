# DATA COLLECTION PROTOCOL
## AI-Based Soldier Survival & Emergency Detection System

---

## 1. EQUIPMENT SETUP

### Sensor Placement (Critical for Consistency)
- **AD8232 ECG**: Standard 3-electrode placement
  - RA (Right Arm): Below right collarbone
  - LA (Left Arm): Below left collarbone  
  - RL (Right Leg / Reference): Lower left ribcage
  - **Ensure electrodes are on CLEAN, DRY skin** — shave hair if needed
  - Use medical-grade ECG adhesive pads (not dry electrodes)

- **MPU6050 IMU**: Strap firmly to the **chest/torso** (sternum area)
  - This gives the best orientation reading for fall detection
  - Must NOT be loose — use elastic band or velcro strap
  - Orientation: X-axis pointing forward, Z-axis pointing up when standing

- **NEO-6M GPS**: Place on the **shoulder or upper arm** (clear sky view)
  - Ensure antenna faces upward
  - Works best outdoors — indoor GPS will be unreliable

- **ESP32**: Central unit strapped to the waist belt or vest pocket
  - Keep wires managed, secure, and away from movement

### Pre-Collection Checklist
- [ ] All sensors powered and responding
- [ ] ECG leads not reporting lead-off (LO+ and LO- both LOW)
- [ ] WiFi connected (check ESP32 Serial Monitor for IP)
- [ ] Python server running and receiving data
- [ ] GPS has fix (may take 1-2 minutes outdoors)
- [ ] Correct LABEL set in data_collection.py
- [ ] Correct SUBJECT_ID set in data_collection.py

---

## 2. DATA COLLECTION SESSIONS

### Target samples per class (depends on your send rate)

Your current ESP32 code sends data every `SEND_INTERVAL = 1000ms` → **~1 sample/second (1 Hz)**.

So roughly:
- **1000 samples** ≈ 16.7 minutes
- **2000 samples** ≈ 33.3 minutes
- **3000 samples** ≈ 50 minutes

### Minimum (college project): 1000–2000 samples per class
### Recommended (more robust): 3000+ samples per class

Tip: your `data_collection.py` prints the live rate like `N=... (x.x/s)` — use that to confirm the real rate on your laptop.

### Session Plan Per Subject

Note: You do NOT need to do every row in this table in one day. You can do 2–4 minute sessions and repeat them across days.

| Session | Label           | Activity                                              | Duration | Rest Between |
|---------|-----------------|-------------------------------------------------------|----------|--------------|
| 1       | normal          | Sit quietly on a chair                                | 5 min    | -            |
| 2       | normal          | Stand still (at attention position)                   | 5 min    | 1 min        |
| 3       | normal          | Lie down on back (simulating guard rest)              | 3 min    | 1 min        |
| 4       | normal          | Walk at normal pace (~4-5 km/h)                       | 5 min    | 2 min        |
| 5       | normal          | Walk at brisk pace (~6 km/h)                          | 5 min    | 2 min        |
| 6       | normal          | Light jog (~8 km/h)                                   | 3 min    | 3 min        |
| 7       | high_exertion   | Sprint / fast run                                     | 2 min    | 3 min        |
| 8       | high_exertion   | Burpees (continuous)                                  | 2 min    | 3 min        |
| 9       | high_exertion   | Pushups + jumping jacks alternating                   | 3 min    | 3 min        |
| 10      | distress        | After high exertion: stop suddenly, remain standing   | 3 min    | 2 min        |
| 11      | distress        | Simulate exhaustion: slow irregular movement, hunched | 3 min    | 2 min        |
| 12      | distress        | Lie down after exertion (elevated HR while still)     | 3 min    | 2 min        |
| 13      | man_down        | Walking → fall forward onto mattress → stay still     | 10 falls | 2 min        |
| 14      | man_down        | Standing → fall sideways onto mattress → stay still   | 10 falls | 2 min        |
| 15      | man_down        | Jogging → fall/collapse → stay still                  | 10 falls | 2 min        |

### For Each Session:
1. Change `CURRENT_LABEL` in data_collection.py
2. Restart the Python server
3. Wait 5 seconds for buffers to initialize
4. BEGIN the activity
5. STOP with CTRL+C when done
6. Note any issues (ECG lead came off, tripped, etc.)

---

## 3. MAN DOWN DATA COLLECTION (Special Protocol)

Man Down is the hardest state to collect because it's an EVENT (fall) followed by a STATE (motionless).

### For Each Fall:
1. Start collecting data with label "man_down"
2. Walk/stand normally for 5 seconds (this transition data is valuable)
3. FALL onto a soft surface (mattress/grass)
4. Remain completely motionless for 15-20 seconds after falling
5. Get up, wait 10 seconds, repeat

### Fall Types to Cover:
- Forward fall (face down)
- Backward fall (on back)  
- Sideways fall (left and right)
- Collapse from standing (knees buckle, crumple down)
- Fall while walking
- Fall while jogging

### Safety: ALWAYS use a mattress or soft grass. DO NOT actually injure yourself.

---

## 4. DISTRESS STATE DETAILS

Distress is the trickiest state because you cannot easily simulate true physiological distress.

### How to Simulate Distress:
1. **Exercise-induced**: Do intense exercise (sprint, burpees) for 3 min → immediately stop 
   and sit/lie down. Your HR will be very high (150+) while your movement is zero — this IS 
   distress-like (the model learns: high HR + low movement = something is wrong).
   
2. **Erratic movement**: Move slowly, irregularly, stumble, simulate being wounded — 
   low/irregular movement + moderate-high HR.

3. **Recovery under load**: After sprinting, walk very slowly as if struggling — 
   high HR + very low movement intensity.

### Key: Distress = mismatch between expected HR and movement level

---

## 5. DATA QUALITY GUIDELINES

### REJECT data if:
- ECG lead-off indicator is ON for more than 30% of a session
- GPS shows no fix for GPS-dependent analysis
- IMU readings are constant (sensor disconnected)
- Subject was adjusting equipment during recording

### KEEP data if:
- Brief ECG dropouts (< 2 seconds) — real-world sensors have this too
- GPS accuracy is low but present — mark it in notes
- Minor movement artifacts in ECG — this is realistic

### Post-Collection Cleaning:
- Remove first 10 seconds of each session (startup artifacts)
- Remove rows where ecg_lead_off = 1 (for ECG features only)
- Check for stuck sensor values (constant readings = hardware issue)
- Verify label correctness by plotting data

---

## 6. MINIMUM DATASET REQUIREMENTS

For a publishable result, you need:

| Requirement          | Minimum     | Recommended  |
|----------------------|-------------|--------------|
| Subjects             | 1 (you)     | 3-5          |
| Samples per class    | 3,000       | 10,000+      |
| Total samples        | 12,000      | 40,000+      |
| Sessions per class   | 3           | 5-10         |
| Collection days      | 2-3 days    | 5+ days      |

### If You Only Have 1 Subject:
This is acceptable for a college project. In your paper, state:
> "Data was collected from a single healthy male subject (age X, weight Xkg, height Xcm). 
> This is acknowledged as a limitation, and future work will validate the model across 
> multiple subjects with diverse physiological profiles."

---

## 7. SESSION LOG TEMPLATE

Fill this out for EVERY collection session:

```
Date: ___________
Time: ___________
Subject: ___________
Label: ___________
Activity: ___________
Duration: ___________
Environment: Indoor / Outdoor
Temperature: ___________
Notes/Issues: ___________
Samples collected: ___________
ECG Quality: Good / Fair / Poor
GPS Fix: Yes / No
```

---

## 8. FEATURES SUMMARY FOR ML MODEL

### Features to USE in ML model (13 features):
1.  bpm             — Heart rate
2.  hrv_sdnn        — Heart rate variability (overall)
3.  hrv_rmssd       — Heart rate variability (short-term)
4.  dynamic_accel   — Body movement intensity
5.  impact          — Sudden acceleration change (jerk)
6.  pitch           — Body tilt angle (lying vs standing)
7.  roll            — Body lateral tilt
8.  movement_var    — Movement consistency over time
9.  gx, gy, gz      — Rotational velocity (3 features)
10. bpm_mean_10s    — Smoothed heart rate trend
11. bpm_std_10s     — Heart rate stability
12. dynamic_accel_mean_5s  — Average movement over 5s
13. dynamic_accel_max_5s   — Peak movement over 5s
14. impact_max_5s          — Peak impact over 5s
15. pitch_mean_5s          — Average body orientation
16. movement_var_mean_5s   — Movement pattern consistency
17. gyro_magnitude_mean_5s — Average rotation intensity

### Features to EXCLUDE from ML model:
- timestamp (metadata)
- session_id, subject_id (metadata)
- ecg_lead_off (quality flag, not a health feature)
- gps_lat, gps_lon, gps_speed, gps_alt (location context only)
- gps_satellites, gps_fix (quality flags)
- smv (redundant with dynamic_accel)

---

## 9. EXPECTED FEATURE RANGES BY STATE

| Feature              | Normal              | High Exertion  | Distress          | Man Down         |
|----------------------|---------------------|----------------|-------------------|------------------|
| BPM                  | 55-130              | 140-190        | 130-190 (or <50)  | Variable/absent  |
| HRV (SDNN)           | 30-100 ms           | 10-30 ms       | 5-20 ms           | Variable         |
| Dynamic Accel        | 0-0.5 g             | 0.3-2.0 g      | 0-0.3 g           | Spike then ~0    |
| Impact               | ~0 to low periodic  | Moderate       | Low/erratic       | HIGH spike → 0   |
| Pitch                | ~0 (upright) or -90 | ~0 (upright)   | Variable          | ~+-90 (lying)    |
| Movement Variance    | Very low to moderate| High           | Low/erratic       | ~0 after fall    |
| Gyro Magnitude       | ~0 to low rhythmic  | Moderate-High  | Low/erratic       | Spike then ~0    |
