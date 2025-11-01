import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

print("--- Synthetic Data Factory (Stage 1) ---")

# --- 1. Configuration ---
# --- Dynamic Path Support ---
if len(sys.argv) > 1:
    REAL_DATA_PATH = sys.argv[1]
    print(f"Using uploaded drifted data: {REAL_DATA_PATH}")
else:
    REAL_DATA_PATH = 'HighDrifted_sample_20k.csv'
    print(f"No file argument passed. Using default: {REAL_DATA_PATH}")
OUTPUT_DATA_PATH = 'synthetic_stats_model_data.csv'
NUM_SAMPLES_TO_GENERATE = 20000  # Generate as many new samples as we have real ones
MODEL_DIR = 'synthetic_model_artifacts'

os.makedirs(MODEL_DIR, exist_ok=True)
gmm_path = os.path.join(MODEL_DIR, 'gmm_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'gmm_scaler.pkl')
discrete_data_path = os.path.join(MODEL_DIR, 'discrete_model_data.csv')

# --- 2. Load Real Data ---
print(f"Loading real data from '{REAL_DATA_PATH}'...")
try:
    df = pd.read_csv(REAL_DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file '{REAL_DATA_PATH}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

print("Data loaded successfully.")

# --- 3. Clean Data ---
print("Cleaning data (handling infinity and NaN values)...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.columns[df.isna().any()].tolist():
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
print("Data cleaning complete.")

# --- 4. Separate Discrete and Continuous Columns ---
# Based on our analysis of your data
discrete_columns = [
    'Destination Port', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Label'
]
discrete_columns = [col for col in discrete_columns if col in df.columns]
continuous_columns = [col for col in df.columns if col not in discrete_columns]

print(f"Identified {len(discrete_columns)} discrete columns.")
print(f"Identified {len(continuous_columns)} continuous columns.")

df_continuous = df[continuous_columns]
df_discrete = df[discrete_columns]

# --- 5. Model Continuous Data (GMM) ---
print("Initializing or loading models...")

# Try to load existing scaler, else create a new one
try:
    scaler = joblib.load(scaler_path)
    print("Loaded existing scaler from artifact.")
except Exception:
    print("Initializing new scaler.")
    scaler = StandardScaler()

# Try to load existing GMM, else create a new one
try:
    gmm = joblib.load(gmm_path)
    print("Loaded existing GMM from artifact.")
    # Set warm_start to True to use existing parameters as initialization
    gmm.warm_start = True
    gmm.verbose = 1
    gmm.max_iter = 200
    gmm.random_state = 42
except Exception:
    print("Initializing new GMM.")
    n_components = 5
    print(f"Using {n_components} components.")
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=42,
                          verbose=1,
                          max_iter=200,
                          warm_start=True) # Use warm_start for potential future re-fits

print("\nScaling continuous data...")
# We ALWAYS re-fit the scaler to the new data's distribution
df_continuous_scaled = scaler.fit_transform(df_continuous)
print("Data scaling complete.")

print("\nTraining Gaussian Mixture Model (GMM)...")
# We ALWAYS re-fit the GMM to the new, scaled data
# If loaded, warm_start=True will use previous params as init
gmm.fit(df_continuous_scaled)
print("GMM training complete.")

# --- 6. Generate New Synthetic Data ---
print(f"\nGenerating {NUM_SAMPLES_TO_GENERATE} new samples...")

# a) Generate new continuous data from the GMM
synthetic_continuous_scaled, _ = gmm.sample(NUM_SAMPLES_TO_GENERATE)
# b) Inverse transform to original scale
synthetic_continuous = scaler.inverse_transform(synthetic_continuous_scaled)
synthetic_continuous_df = pd.DataFrame(synthetic_continuous, columns=continuous_columns)

# c) Generate new discrete data by sampling from original
synthetic_discrete_df = df_discrete.sample(n=NUM_SAMPLES_TO_GENERATE,
                                           replace=True).reset_index(drop=True)

# d) Combine and save
synthetic_data = pd.concat([synthetic_continuous_df, synthetic_discrete_df], axis=1)
synthetic_data = synthetic_data[df.columns]  # Ensure original column order

synthetic_data.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"Successfully saved synthetic data to '{OUTPUT_DATA_PATH}'.")

# --- 7. Save Models ---
# Overwrite the models with the newly trained versions
joblib.dump(gmm, gmm_path)
joblib.dump(scaler, scaler_path)
df_discrete.to_csv(discrete_data_path, index=False)
print(f"GMM models saved/updated in '{MODEL_DIR}'.")
print("--- Synthetic Data Factory (Stage 1) Complete ---")

