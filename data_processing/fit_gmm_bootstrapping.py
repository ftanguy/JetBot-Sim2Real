import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.utils import resample
import os

# --- 1. DATA LOADING ---
filename = "identification_data_1(in).csv"
csv_path = None

search_paths = [".", "/kaggle/input", "/content"]
for search_root in search_paths:
    if os.path.exists(search_root):
        for root, dirs, files in os.walk(search_root):
            if filename in files:
                csv_path = os.path.join(root, filename)
                break
    if csv_path: break

# --- 2. LOAD DATA ---
try:
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"Data Loaded: {len(df)} rows from {csv_path}")
    else:
        print("File not found. Using dummy data for structure verification.")
        df = pd.DataFrame({'ul': [], 'ur': [], 'omegaL_measured': [], 'omegaR_measured': []})
except Exception as e:
    print(f"Error loading data: {e}")
    df = None

# --- 3. HELPER FUNCTIONS ---

def clean_physics_data(u, w):
    """Removes physical impossibilities (sign mismatch) and tiny noise."""
    mask_sign = (np.sign(u) == np.sign(w)) | (np.abs(w) < 0.1)
    mask_magnitude = np.abs(u) > 0.01
    valid_mask = mask_sign & mask_magnitude
    return u[valid_mask], w[valid_mask]

def get_distribution_params(u_group, w_group, mode, n_boot=500):
    """Bootstraps the data to get Mean and Std Dev of parameters."""
    if len(u_group) < 10: return None
    
    alphas, deadzones = [], []
    def model(x, a, d): return np.maximum(0, a * (x - d))
    
    # Physics-informed bounds
    if mode == 'puddle':
        bounds, p0 = ([4.0, 0.0], [25.0, 0.6]), [10.0, 0.4]
    else: # dry
        bounds, p0 = ([30.0, 0.0], [70.0, 0.1]), [40.0, 0.05]
        
    for _ in range(n_boot):
        u_res, w_res = resample(u_group, w_group)
        
        # Stall Filter (Only for Puddle fitting)
        if mode == 'puddle':
            moving_mask = w_res > 0.5
            if np.sum(moving_mask) > 5:
                u_fit, w_fit = u_res[moving_mask], w_res[moving_mask]
            else:
                u_fit, w_fit = u_res, w_res
        else:
            u_fit, w_fit = u_res, w_res
            
        try:
            popt, _ = curve_fit(model, u_fit, w_fit, p0=p0, bounds=bounds)
            alphas.append(popt[0])
            deadzones.append(popt[1])
        except: continue
            
    if not alphas: return None
    
    return {
        'alpha_mean': np.mean(alphas),
        'alpha_std': np.std(alphas),
        'dz_mean': np.mean(deadzones),
        'dz_std': np.std(deadzones)
    }

def analyze_and_plot(u_raw, w_raw, ax, title):
    u_data, w_data = clean_physics_data(u_raw, w_raw)
    if len(u_data) < 10: return

    u_abs, w_abs = np.abs(u_data), np.abs(w_data)
    def model(x, a, d): return np.maximum(0, a * (x - d))

    # --- A. INITIAL CLASSIFICATION (Guess) ---
    y_dry_guess = model(u_abs, 35.0, 0.05)
    y_puddle_guess = model(u_abs, 5.0, 0.5)
    labels = np.where((w_abs - y_dry_guess)**2 < (w_abs - y_puddle_guess)**2, 1, 0)
    
    # --- B. REFINE CLASSIFICATION (Fit once to get better lines) ---
    # We fit based on the initial guess groups to get the "Real" lines
    def single_fit(u, w, m):
        if m == 'puddle':
            mask = w > 0.5
            if np.sum(mask) > 5: u, w = u[mask], w[mask]
            bounds = ([4.0, 0.0], [25.0, 0.6])
            p0 = [10.0, 0.4]
        else:
            bounds = ([30.0, 0.0], [70.0, 0.1])
            p0 = [40.0, 0.05]
        try: return curve_fit(model, u, w, p0=p0, bounds=bounds)[0]
        except: return None

    popt_p = single_fit(u_abs[labels==0], w_abs[labels==0], 'puddle')
    popt_d = single_fit(u_abs[labels==1], w_abs[labels==1], 'dry')
    
    # Use these better lines to re-classify strictly by distance
    if popt_p is not None and popt_d is not None:
        y_p = model(u_abs, *popt_p)
        y_d = model(u_abs, *popt_d)
        
        # --- PURE MATH CLASSIFICATION (No Physics Override) ---
        # Assign to whichever line reduces squared error
        labels = np.where((w_abs - y_d)**2 < (w_abs - y_p)**2, 1, 0)
        
    # --- C. BOOTSTRAPPING (Get Mean ± Std) ---
    print(f"Bootstrapping {title} distributions...")
    stats_p = get_distribution_params(u_abs[labels==0], w_abs[labels==0], 'puddle')
    stats_d = get_distribution_params(u_abs[labels==1], w_abs[labels==1], 'dry')
    
    # --- D. REPORTING ---
    print(f"\n--- {title} ---")
    if stats_p:
        print(f"   PUDDLE (Pink):")
        print(f"     Alpha:    {stats_p['alpha_mean']:.2f} ± {stats_p['alpha_std']:.2f}")
        print(f"     Deadzone: {stats_p['dz_mean']:.3f} ± {stats_p['dz_std']:.3f}")
        print(f"     Weight:   {np.mean(labels==0):.2f}")
        
    if stats_d:
        print(f"   DRY (Green):")
        print(f"     Alpha:    {stats_d['alpha_mean']:.2f} ± {stats_d['alpha_std']:.2f}")
        print(f"     Deadzone: {stats_d['dz_mean']:.3f} ± {stats_d['dz_std']:.3f}")
        print(f"     Weight:   {np.mean(labels==1):.2f}")

    # --- E. PLOTTING ---
    # Apply "Influence Filter" for plotting (hide stalled puddle points so we see the trend)
    plot_mask_p = (labels == 0) & (w_abs > 0.5)
    plot_mask_d = (labels == 1)
    
    ax.scatter(u_data[plot_mask_p], w_data[plot_mask_p], s=15, color='deeppink', alpha=0.4, label='_nolegend_')
    ax.scatter(u_data[plot_mask_d], w_data[plot_mask_d], s=15, color='lime', alpha=0.4, label='_nolegend_')

    x_grid = np.linspace(min(u_data), max(u_data), 100)
    sign_grid = np.sign(x_grid)
    
    if stats_p:
        y_p = sign_grid * model(np.abs(x_grid), stats_p['alpha_mean'], stats_p['dz_mean'])
        ax.plot(x_grid, y_p, color='deeppink', linewidth=3, label='Puddle Mean Fit')
    if stats_d:
        y_d = sign_grid * model(np.abs(x_grid), stats_d['alpha_mean'], stats_d['dz_mean'])
        ax.plot(x_grid, y_d, color='lime', linewidth=3, label='Dry Mean Fit')

# --- MAIN EXECUTION ---
if df is not None and not df.empty:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Concatenate Left and Right data for a combined analysis
    u_all = np.concatenate([df['ur'].values, df['ul'].values])
    w_all = np.concatenate([df['omegaR_measured'].values, df['omegaL_measured'].values])

    print("="*20 + " COMBINED WHEELS " + "="*20)
    analyze_and_plot(u_all, w_all, ax, "Combined Data")

    ax.set_title("Combined Wheels Analysis (Pure Residual Classification)")
    ax.set_xlabel("Input u")
    ax.set_ylabel("Wheel Speed")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No data available to plot.")