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
ORIGINAL_DATA_PATH = "reference_sample.csv"

if len(sys.argv) > 1:
    DRIFTED_DATA_PATH = sys.argv[1]
    print(f"Using uploaded drifted dataset: {DRIFTED_DATA_PATH}")
else:
    DRIFTED_DATA_PATH = "HighDriften_sample_20k.csv"
    print(f"Using default drifted dataset: {DRIFTED_DATA_PATH}")

SYNTHETIC_DATA_PATH = "synthetic_stats_model_data.csv"

MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.pkl")

min_samples_for_smote = 6
os.makedirs(MODEL_DIR, exist_ok=True)

# === 2. Load and Combine Data ===
def load_and_combine_data():
    df_original, df_drifted, df_synthetic = None, None, None

    try:
        df_original = pd.read_csv(ORIGINAL_DATA_PATH, low_memory=False)
    except FileNotFoundError:
        pass
    
    try:
        df_drifted = pd.read_csv(DRIFTED_DATA_PATH, low_memory=False)
    except FileNotFoundError:
        pass
    
    try:
        df_synthetic = pd.read_csv(SYNTHETIC_DATA_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Synthetic data '{SYNTHETIC_DATA_PATH}' not found.")
        return None

    all_dfs = [df for df in [df_original, df_drifted, df_synthetic] if df is not None]
    if not all_dfs:
        print("Error: No data available for training.")
        return None

    return pd.concat(all_dfs, ignore_index=True)

# === 3. Clean and Prepare Data ===
def clean_data(df):
    df.columns = df.columns.str.strip()
    canonical_label_name = 'Label'

    all_label_cols = [col for col in df.columns if col.lower() == 'label']
    if not all_label_cols:
        print("Error: 'Label' column not found.")
        return None, None

    labels_1d = df[all_label_cols[0]]
    if isinstance(labels_1d, pd.DataFrame):
        labels_1d = labels_1d.iloc[:, 0]

    feature_cols = [col for col in df.columns if col.lower() != 'label']
    df = df[feature_cols].copy()
    df[canonical_label_name] = labels_1d
    df.dropna(subset=[canonical_label_name], inplace=True)

    df[canonical_label_name] = pd.Categorical(df[canonical_label_name]).codes
    class_counts = df[canonical_label_name].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_for_smote].index
    df = df[df[canonical_label_name].isin(valid_classes)]
    df[canonical_label_name] = pd.Categorical(df[canonical_label_name]).codes

    X = df.drop(columns=[canonical_label_name])
    y = df[canonical_label_name]

    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)

    return X, y

# === 4. Main Training Pipeline ===
def main():
    df = load_and_combine_data()
    if df is None:
        sys.exit(1)

    X, y = clean_data(df)
    if X is None or y is None:
        sys.exit(1)

    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    except Exception:
        X_train_resampled, y_train_resampled = X_train, y_train

    try:
        model = joblib.load(MODEL_PATH)
        model.use_label_encoder = False
        model.eval_metric = 'mlogloss'
        model.random_state = 42
    except Exception:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Retrained Model Accuracy: {acc:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selector, SELECTOR_PATH)
    joblib.dump({"accuracy": acc, "confusion_matrix": cm, "report": report_dict}, METRICS_PATH)

    print("Model retraining complete.")

if __name__ == "__main__":
    main()
