import pandas as pd
import numpy as np
import time

# Set random seed for reproducibility
np.random.seed(42)

# Features to generate as specified
ML_FEATURES = [
    "bpm",
    "hrv_sdnn",
    "hrv_rmssd",
    "dynamic_accel",
    "impact",
    "pitch",
    "roll",
    "gx",
    "gy",
    "gz",
    "movement_var",
    "bpm_mean_10s",
    "bpm_std_10s",
    "dynamic_accel_mean_5s",
    "dynamic_accel_max_5s",
    "impact_max_5s",
    "pitch_mean_5s",
    "movement_var_mean_5s",
    "gyro_magnitude_mean_5s",
]

def generate_session_data(session_id, label, n_samples):
    """Generates a synthetic time-series session for a given label."""
    # Base timestamp
    start_time = time.time() - np.random.randint(0, 100000)
    timestamps = [start_time + i for i in range(n_samples)]
    
    # Generate base features depending on the state
    if label == 'normal':
        # Resting / routine patrol. Messy real-world distributions
        bpm = np.random.normal(80, 15, n_samples)
        hrv_sdnn = np.random.normal(40, 15, n_samples)
        hrv_rmssd = np.random.normal(30, 12, n_samples)
        dynamic_accel = np.random.normal(0.3, 0.3, n_samples)
        impact = np.random.exponential(0.1, n_samples)
        pitch = np.random.normal(20, 15, n_samples)
        roll = np.random.normal(0, 15, n_samples)
        gx = np.random.normal(0, 10, n_samples)
        gy = np.random.normal(0, 10, n_samples)
        gz = np.random.normal(0, 10, n_samples)
        movement_var = np.random.normal(0.1, 0.05, n_samples)
    elif label == 'high_exertion':
        # Running / combat activity. Can have moments of rest (overlap)
        bpm = np.random.normal(145, 25, n_samples)
        hrv_sdnn = np.random.normal(30, 10, n_samples)
        hrv_rmssd = np.random.normal(20, 8, n_samples)
        dynamic_accel = np.random.normal(1.5, 0.8, n_samples)
        impact = np.random.exponential(0.8, n_samples)
        pitch = np.random.normal(35, 20, n_samples)
        roll = np.random.normal(0, 20, n_samples)
        gx = np.random.normal(0, 40, n_samples)
        gy = np.random.normal(0, 40, n_samples)
        gz = np.random.normal(0, 40, n_samples)
        movement_var = np.random.normal(0.6, 0.2, n_samples)
    elif label == 'man_down':
        # Unconscious / incapacitated. Noise simulates erratic sensors or partial falls
        bpm = np.random.normal(45, 30, n_samples)
        hrv_sdnn = np.random.normal(15, 10, n_samples)
        hrv_rmssd = np.random.normal(10, 5, n_samples)
        dynamic_accel = np.random.normal(0.1, 0.15, n_samples)
        impact = np.random.exponential(0.02, n_samples) # random small bumps
        pitch = np.random.normal(75, 20, n_samples) # Not always perfectly 90
        roll = np.random.normal(80, 25, n_samples)
        gx = np.random.normal(0, 5, n_samples)
        gy = np.random.normal(0, 5, n_samples)
        gz = np.random.normal(0, 5, n_samples)
        movement_var = np.random.normal(0.02, 0.01, n_samples)
    else:
        raise ValueError(f"Unknown label: {label}")
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'session_id': [session_id] * n_samples,
        'bpm': np.clip(bpm, 0, 220),
        'hrv_sdnn': np.clip(hrv_sdnn, 0, None),
        'hrv_rmssd': np.clip(hrv_rmssd, 0, None),
        'dynamic_accel': np.clip(dynamic_accel, 0, None),
        'impact': np.clip(impact, 0, None),
        'pitch': pitch,
        'roll': roll,
        'gx': gx,
        'gy': gy,
        'gz': gz,
        'movement_var': np.clip(movement_var, 0, None),
        'label': [label] * n_samples
    })
    
    # Calculate windowed features
    df['bpm_mean_10s'] = df['bpm'].rolling(window=10, min_periods=1).mean()
    df['bpm_std_10s'] = df['bpm'].rolling(window=10, min_periods=1).std().fillna(0)
    
    df['dynamic_accel_mean_5s'] = df['dynamic_accel'].rolling(window=5, min_periods=1).mean()
    df['dynamic_accel_max_5s'] = df['dynamic_accel'].rolling(window=5, min_periods=1).max()
    
    df['impact_max_5s'] = df['impact'].rolling(window=5, min_periods=1).max()
    
    df['pitch_mean_5s'] = df['pitch'].rolling(window=5, min_periods=1).mean()
    
    df['movement_var_mean_5s'] = df['movement_var'].rolling(window=5, min_periods=1).mean()
    
    gyro_mag = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    df['gyro_magnitude_mean_5s'] = gyro_mag.rolling(window=5, min_periods=1).mean()
    
    # Fill any remaining NaNs just in case
    df = df.fillna(0)
    
    return df

def main():
    print("Generating synthetic data for soldier survival detection...")
    
    # We drop to the 3 recommended classes
    labels = ['normal', 'high_exertion', 'man_down']
    
    # Generate 20 sessions per class, each 5 minutes long (300 rows at 1 Hz)
    sessions_per_label = 20
    samples_per_session = 300 
    
    all_dfs = []
    session_counter = 1
    
    for label in labels:
        print(f"Generating data for class: {label}...")
        for _ in range(sessions_per_label):
            session_id = f"SYNTH_{session_counter:03d}"
            df_session = generate_session_data(session_id, label, samples_per_session)
            all_dfs.append(df_session)
            session_counter += 1
            
    # Combine all sessions
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure column order matches expected + session info
    cols = ['timestamp', 'session_id'] + ML_FEATURES + ['label']
    final_df = final_df[cols]
    
    # Output to soldier_data.csv to seamlessly integrate with model_training.py
    output_file = "soldier_data.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully generated {len(final_df)} rows of synthetic data.")
    print(f"Data saved to '{output_file}'.")
    print("\nClass distribution:")
    print(final_df['label'].value_counts())
    print("\nData is ready to be used with model_training.py!")

if __name__ == "__main__":
    main()
