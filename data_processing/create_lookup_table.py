import numpy as np
import pandas as pd
import os

# --- CONFIG ---
BIN_SIZE = 0.05  # Granularity of input (0.0, 0.05, 0.10...)
MAX_SAMPLES = 2000 # Samples per bin to keep

# --- 1. LOAD DATA ---
filename = "identification_data_1(in).csv"
csv_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    if filename in files:
        csv_path = os.path.join(root, filename)
        break

df = pd.read_csv(csv_path)

# Combine Left/Right
u_all = np.concatenate([df['ur'].values, df['ul'].values])
w_all = np.concatenate([df['omegaR_measured'].values, df['omegaL_measured'].values])

# --- 2. CREATE LOOKUP TABLE ---
# We map input range [-1.0, 1.0] to indices
# Formula: index = (u + 1.0) / BIN_SIZE
num_bins = int(2.0 / BIN_SIZE) + 1
physics_table = np.zeros((num_bins, MAX_SAMPLES))

print(f"Generating Table with {num_bins} bins...")

for i in range(num_bins):
    # Determine input range for this bin
    u_low = -1.0 + (i * BIN_SIZE)
    u_high = u_low + BIN_SIZE
    
    # Find real data points in this range
    mask = (u_all >= u_low) & (u_all < u_high)
    speeds = w_all[mask]
    
    if len(speeds) > 0:
        # If we have data, randomly fill the row with it
        # We start by filling with available data
        take_n = min(len(speeds), MAX_SAMPLES)
        physics_table[i, :take_n] = speeds[:take_n]
        
        # If we need more to fill the row to MAX_SAMPLES, resample with replacement
        if take_n < MAX_SAMPLES:
            needed = MAX_SAMPLES - take_n
            physics_table[i, take_n:] = np.random.choice(speeds, needed)
    else:
        # GAP FILLING: If no data exists for this input (e.g. u=0.9)
        # We assume linear interpolation from Dry Alpha=50 behavior
        # Speed = 50 * (u - 0.1)
        # This prevents the robot from getting "NaN" reward if it explores new areas
        center_u = (u_low + u_high) / 2
        # Use simple model: 50 * (u - 0.1)
        simulated_speed = 38.0 * (np.abs(center_u) - 0.05) # ALPHA = 38 AND DEADZONE = 0.05
        simulated_speed = np.maximum(0, simulated_speed) * np.sign(center_u)
        physics_table[i, :] = simulated_speed

# --- 3. SAVE ---
np.save("physics_lookup.npy", physics_table)
print("Success! Saved 'physics_lookup.npy'.")
print("Move this file to the same folder as your training script.")