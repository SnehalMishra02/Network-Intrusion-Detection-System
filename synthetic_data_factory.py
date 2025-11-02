import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

print("--- Synthetic Data Factory (Stage 1) ---")

# --- 1. Configuration ---
if len(sys.argv) > 1:
    REAL_DATA_PATH = sys.argv[1]
    print(f"Using uploaded drifted data: {REAL_DATA_PATH}")
else:
    REAL_DATA_PATH = 'HighDrifted_sample_20k.csv'
    print(f"Using default drifted data: {REAL_DATA_PATH}")

OUTPUT_DATA_PATH = 'synthetic_stats_model_data.csv'
NUM_SAMPLES_TO_GENERATE = 20000
MODEL_DIR = 'synthetic_model_artifacts'

os.makedirs(MODEL_DIR, exist_ok=True)
gmm_path = os.path.join(MODEL_DIR, 'gmm_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'gmm_scaler.pkl')
discrete_data_path = os.path.join(MODEL_DIR, 'discrete_model_data.csv')

# --- 2. Load Real Data ---
try:
    df = pd.read_csv(REAL_DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file '{REAL_DATA_PATH}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

# --- 3. Clean Data ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.columns[df.isna().any()].tolist():
    df[col].fillna(df[col].median(), inplace=True)

# --- 4. Separate Discrete and Continuous Columns ---
discrete_columns = [
    'Destination Port', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Label'
]
discrete_columns = [col for col in discrete_columns if col in df.columns]
continuous_columns = [col for col in df.columns if col not in discrete_columns]

df_continuous = df[continuous_columns]
df_discrete = df[discrete_columns]

# --- 5. Model Continuous Data (GMM) ---
try:
    scaler = joblib.load(scaler_path)
except Exception:
    scaler = StandardScaler()

try:
    gmm = joblib.load(gmm_path)
    gmm.warm_start = True
    gmm.verbose = 0
    gmm.max_iter = 200
    gmm.random_state = 42
except Exception:
    gmm = GaussianMixture(
        n_components=5,
        covariance_type='full',
        random_state=42,
        verbose=0,
        max_iter=200,
        warm_start=True
    )

df_continuous_scaled = scaler.fit_transform(df_continuous)
gmm.fit(df_continuous_scaled)

# --- 6. Generate New Synthetic Data ---
synthetic_continuous_scaled, _ = gmm.sample(NUM_SAMPLES_TO_GENERATE)
synthetic_continuous = scaler.inverse_transform(synthetic_continuous_scaled)
synthetic_continuous_df = pd.DataFrame(synthetic_continuous, columns=continuous_columns)

synthetic_discrete_df = df_discrete.sample(
    n=NUM_SAMPLES_TO_GENERATE,
    replace=True
).reset_index(drop=True)

synthetic_data = pd.concat([synthetic_continuous_df, synthetic_discrete_df], axis=1)
synthetic_data = synthetic_data[df.columns]
synthetic_data.to_csv(OUTPUT_DATA_PATH, index=False)

# --- 7. Save Models ---
joblib.dump(gmm, gmm_path)
joblib.dump(scaler, scaler_path)
df_discrete.to_csv(discrete_data_path, index=False)

print("Synthetic data generation complete.")
