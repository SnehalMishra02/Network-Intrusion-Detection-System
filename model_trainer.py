# === Import libraries ===
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import joblib
import os
import sys

print("--- Model Trainer (Stage 2) ---")

# === 1. Configuration ===
# Paths to the data (must match our MLOps pipeline)
ORIGINAL_DATA_PATH = "reference_sample.csv"
# --- Dynamic Path Support ---
if len(sys.argv) > 1:
    DRIFTED_DATA_PATH = sys.argv[1]
    print(f"Using uploaded drifted dataset: {DRIFTED_DATA_PATH}")
else:
    DRIFTED_DATA_PATH = "HighDriften_sample_20k.csv"  # fallback default
    print(f"No argument provided. Using default: {DRIFTED_DATA_PATH}")

SYNTHETIC_DATA_PATH = "synthetic_stats_model_data.csv"

# Paths to the model artifacts (must match app.py)
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.pkl") # For streamlit viz

# --- FIX: Define the missing variable ---
# SMOTE needs at least 6 samples per class in the test split to work.
min_samples_for_smote = 6 
# --- End of fix ---

os.makedirs(MODEL_DIR, exist_ok=True)

# === 2. Load and Combine Data ===
def load_and_combine_data():
    """Loads original, drifted, and synthetic data, then combines them."""
    print("Loading datasets...")
    df_original, df_drifted, df_synthetic = None, None, None

    try:
        df_original = pd.read_csv(ORIGINAL_DATA_PATH, low_memory=False)
        print(f"Loaded {len(df_original)} rows from '{ORIGINAL_DATA_PATH}'")
    except FileNotFoundError:
        print(f"Warning: '{ORIGINAL_DATA_PATH}' not found. Skipping.")
    
    try:
        df_drifted = pd.read_csv(DRIFTED_DATA_PATH, low_memory=False)
        print(f"Loaded {len(df_drifted)} rows from '{DRIFTED_DATA_PATH}'")
    except FileNotFoundError:
        print(f"Warning: '{DRIFTED_DATA_PATH}' not found. Skipping.")
    
    try:
        df_synthetic = pd.read_csv(SYNTHETIC_DATA_PATH, low_memory=False)
        print(f"Loaded {len(df_synthetic)} rows from '{SYNTHETIC_DATA_PATH}'")
    except FileNotFoundError:
        print(f"Error: Synthetic data '{SYNTHETIC_DATA_PATH}' not found. This is required for retraining.")
        return None

    all_dfs = [df for df in [df_original, df_drifted, df_synthetic] if df is not None]
    
    if not all_dfs:
        print("Error: No data found to train on.")
        return None

    print("Combining datasets...")
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows for retraining: {len(df_combined)}")
    return df_combined

# === 3. Clean and Prepare Data ===
def clean_data(df):
    """Cleans data, converts labels to codes, handles NaNs, and samples."""
    print("Cleaning data...")
    df.columns = df.columns.str.strip()
    
    canonical_label_name = 'Label' # The name we will enforce

    # --- FIX for 'Grouper not 1-dimensional' error ---
    # This error means 'Label' is a DataFrame due to duplicate columns.
    # We must explicitly separate features and the one *true* label column.
    
    all_label_cols = [col for col in df.columns if col.lower() == 'label']
    if not all_label_cols:
        print("Error: 'Label' column not found.")
        return None, None

    if len(all_label_cols) > 1:
        print(f"Warning: Multiple 'Label' columns found: {all_label_cols}. Consolidating.")

    # 1. Get the 1D Series of labels
    labels_1d = df[all_label_cols[0]] # Get the first 'Label' column
    if isinstance(labels_1d, pd.DataFrame):
        labels_1d = labels_1d.iloc[:, 0] # Flatten if it's still a DataFrame
    
    # 2. Get all feature columns (everything NOT named 'Label')
    feature_cols = [col for col in df.columns if col.lower() != 'label']
    
    # 3. Re-create the DataFrame cleanly
    df = df[feature_cols].copy()
    df[canonical_label_name] = labels_1d
    # --- End of fix ---

    # Now we are *guaranteed* df[canonical_label_name] is a 1D Series
    
    # Drop rows where Label is missing
    df.dropna(subset=[canonical_label_name], inplace=True)

    print("Converting labels to categorical codes (step 1)...")
    df[canonical_label_name] = pd.Categorical(df[canonical_label_name]).codes

    # --- Keep only classes with enough samples for SMOTE ---
    class_counts = df[canonical_label_name].value_counts() # This will now work
    valid_classes = class_counts[class_counts >= min_samples_for_smote].index
    df = df[df[canonical_label_name].isin(valid_classes)]

    print("Re-encoding labels after sampling (step 2)...")
    df[canonical_label_name] = pd.Categorical(df[canonical_label_name]).codes

    # --- Separate features and label ---
    X = df.drop(columns=[canonical_label_name])
    y = df[canonical_label_name]

    # --- Sanitize feature data ---
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)

    print(f"Cleaned data shape: {X.shape}, unique labels: {len(np.unique(y))}")
    return X, y



# === 4. Main Training Pipeline ===
def main():
    df = load_and_combine_data()
    if df is None:
        sys.exit(1)

    X, y = clean_data(df)
    if X is None or y is None:
        sys.exit(1)

    # --- Load or Initialize Selector ---
    print("Initializing new VarianceThreshold selector...")
    # ALWAYS initialize a new selector to avoid scikit-learn version conflicts.
    # We are re-fitting it from scratch anyway.
    selector = VarianceThreshold(threshold=0.01)

    # ALWAYS re-fit the selector to the new data
    print("Fitting new selector...")
    X_selected = selector.fit_transform(X)

    # --- Train-test split ---
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # --- Load or Initialize Scaler ---
    print("Initializing new StandardScaler...")
    # ALWAYS initialize a new scaler to avoid scikit-learn version conflicts.
    # We are re-fitting it from scratch anyway.
    scaler = StandardScaler()

    # ALWAYS re-fit the scaler to the new training data
    print("Fitting new scaler...")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Use transform, not fit_transform

    # --- SMOTE for class imbalance ---
    print("Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTE failed (likely due to low class counts): {e}. Training without SMT.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # --- Load or Initialize XGBoost Model ---
    print("Loading or initializing XGBoost model...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded existing XGBoost model artifact.")
        # Ensure params are set for retraining
        model.use_label_encoder = False
        model.eval_metric = 'mlogloss'
        model.random_state = 42
    except Exception:
        print("Initializing new XGBoost model.")
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    # --- Train XGBoost classifier ---
    print("Starting model training...")
    # This will retrain the model (whether loaded or new) on the new resampled data
    model.fit(X_train_resampled, y_train_resampled)
    print("Model training complete.")

    # --- Make predictions and evaluate ---
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nRetrained Model Accuracy: {acc:.4f}")
    print("\nRetrained Classification Report:\n", report_str)

    # --- Save model, scaler, selector, and metrics ---
    print("Saving all artifacts...")
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")

    joblib.dump(selector, SELECTOR_PATH)
    print(f"Selector saved to: {SELECTOR_PATH}")
    
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report_dict
    }
    joblib.dump(metrics, METRICS_PATH)
    print(f"Model metrics saved to: {METRICS_PATH}")
    
    print("--- Model Trainer (Stage 2) Complete ---")

if __name__ == "__main__":
    main()

