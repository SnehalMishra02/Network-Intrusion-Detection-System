# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from river import compose, preprocessing, linear_model, metrics, drift
from river import tree, ensemble, neighbors, naive_bayes

# NEW: Deep model predictor
from model_dl.predict_nn import NNPredictor

# App Configuration
st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

# Paths to Model Artifacts
MODEL_PATH = "./model/xgb_model.pkl"
SCALER_PATH = "./model/scaler.pkl"
SELECTOR_PATH = "./model/selector.pkl"
REFERENCE_SAMPLE_PATH = "./reference_sample.csv"
LABEL_MAPPING_PATH = "./model/label_mapping.pkl"
ONLINE_MODEL_PATH = "./model/online_model.pkl"

# NEW: DL artifacts directory (created by your NN trainer)
DL_ARTIFACTS_DIR = "./artifacts"  # must contain nn.pt, scaler.pkl, nn_meta.json

# ---------------------- Loaders ----------------------

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)
        reference = pd.read_csv(REFERENCE_SAMPLE_PATH)
        label_mapping = joblib.load(LABEL_MAPPING_PATH) if os.path.exists(LABEL_MAPPING_PATH) else None
        online_model = joblib.load(ONLINE_MODEL_PATH) if os.path.exists(ONLINE_MODEL_PATH) else compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.HoeffdingTreeClassifier()
        )
        return model, scaler, selector, reference, label_mapping, online_model
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, None, None

# NEW: cache-load the deep model
@st.cache_resource
def load_nn():
    try:
        return NNPredictor(artifacts_dir=DL_ARTIFACTS_DIR)
    except Exception as e:
        st.warning(f"Deep model not available: {e}")
        return None

# Initialize Models and Metrics
model, scaler, selector, reference_df, label_mapping, online_model = load_artifacts()
online_metrics = metrics.ClassificationReport()
drift_detector = drift.ADWIN()
inverse_map = {v: k for k, v in label_mapping.items()} if label_mapping else None
nn = load_nn()

# ---------------------- Utils ----------------------

# Drift Detection Function
def detect_drift(reference_data, incoming_data, feature_names, threshold=0.05):
    drifted_features = []
    for col in feature_names:
        if col in incoming_data.columns:
            stat, p = ks_2samp(reference_data[col].dropna(), incoming_data[col].dropna())
            if p < threshold:
                drifted_features.append((col, p))
    return drifted_features

# Prepare Data for Online Learning
def prepare_river_sample(row, feature_names, target_col=None):
    features = {col: row[col] for col in feature_names if col in row}
    if target_col and target_col in row:
        return features, row[target_col]
    return features

# ---------------------- UI ----------------------

# App Title and Description
st.title("Network Intrusion Detection System")
st.markdown("""
Upload a CSV file of network traffic to classify entries, detect data drift, and evaluate model performance.
This system uses a batch-trained XGBoost model, supports online learning with Hoeffding Tree, and now includes a Deep Learning (MLP) baseline.
""")

# File Upload and Input
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
ground_truth_col = st.text_input("Enter the ground truth column name (if available):")
online_learning_toggle = st.checkbox("Enable Online Learning", value=True)
use_nn = st.checkbox("Enable Deep Learning (MLP)", value=True)

# ---------------------- Main ----------------------

if uploaded_file and model and scaler and selector and reference_df is not None:
    try:
        # Load and Display Uploaded Data
        df = pd.read_csv(uploaded_file)
        st.subheader("1. Uploaded Data Sample")
        st.dataframe(df.head())

        if ground_truth_col and ground_truth_col not in df.columns:
            st.error(f"Ground truth column '{ground_truth_col}' not found.")
            st.stop()

        # Preprocess Data
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_df.fillna(numeric_df.mean(numeric_only=True), inplace=True)

        # Drift Detection
        st.subheader("2. Drift Detection")
        common_cols = list(set(reference_df.columns).intersection(numeric_df.columns))
        reference_clean = reference_df[common_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        incoming_clean = numeric_df[common_cols].copy()
        drifted = detect_drift(reference_clean, incoming_clean, common_cols)

        if drifted:
            st.warning(f"Drift detected in {len(drifted)} feature(s):")
            for col, p in drifted[:10]:
                st.write(f"- {col} (p = {p:.5f})")
        else:
            st.success("No significant feature drift detected.")

        # Feature Selection and Scaling for Batch Model
        feature_names = selector.get_feature_names_out().tolist()
        # IMPORTANT: ensure we don't accidentally include the ground-truth column as a feature
        if ground_truth_col:
            feature_names = [c for c in feature_names if c != ground_truth_col]
        selected_df = numeric_df.reindex(columns=feature_names, fill_value=0.0)
        scaled = scaler.transform(selected_df.astype("float64"))
        batch_predictions = model.predict(scaled)

        # Online Learning and Predictions
        online_predictions = []
        batch_correct = 0
        online_correct = 0
        total_samples = 0

        for idx, row in df.iterrows():
            river_sample = prepare_river_sample(row, feature_names, ground_truth_col if ground_truth_col else None)

            if ground_truth_col and ground_truth_col in row:
                x, y = river_sample
                has_ground_truth = True
                if label_mapping:
                    # map string label -> numeric id expected by your models
                    y = {v: k for k, v in label_mapping.items()}.get(y, y)
            else:
                x = river_sample
                y = None
                has_ground_truth = False

            online_pred = online_model.predict_one(x)
            online_pred_numeric = label_mapping[online_pred] if label_mapping and online_pred in label_mapping else -1
            online_predictions.append(online_pred_numeric)

            if has_ground_truth:
                drift_detector.update(int(online_pred != y))
                if drift_detector.drift_detected:
                    st.warning("Concept drift detected in online model!")

            if online_learning_toggle and has_ground_truth:
                online_model.learn_one(x, y)
                if y is not None and online_pred is not None:
                    online_metrics.update(y, online_pred)
                if batch_predictions[idx] == label_mapping.get(y, -1):
                    batch_correct += 1
                if online_pred == y:
                    online_correct += 1
                total_samples += 1

        # Persist updated online model
        joblib.dump(online_model, ONLINE_MODEL_PATH)

        # ---------------------- Deep Learning Predictions (NEW) ----------------------
        st.subheader("5. Deep Learning Model (MLP)")
        nn_pred_numeric = None
        nn_labels_for_table = None
        nn_conf = None

        if use_nn and nn is not None:
            df_for_nn = df.drop(columns=[ground_truth_col], errors="ignore")
            nn_pred_str, nn_conf = nn.predict(df_for_nn, label_col=None)

            def map_nn_to_eval_id(lbl: str) -> int:
                s = str(lbl)
                if label_mapping and s in label_mapping:
                    return int(label_mapping[s])
                try:
                    return int(s)
                except Exception:
                    return -1

            nn_pred_numeric = [map_nn_to_eval_id(s) for s in nn_pred_str]

            if inverse_map and all(pid in inverse_map for pid in nn_pred_numeric):
                nn_labels_for_table = [inverse_map[pid] for pid in nn_pred_numeric]
            else:
                nn_labels_for_table = nn_pred_str

            st.success("Deep model predictions ready.")

            with st.expander("Debug: NN vs ground truth (first 10)"):
                st.write("NN raw string preds:", nn_pred_str[:10])
                st.write("NN mapped numeric preds:", nn_pred_numeric[:10])
                if ground_truth_col and ground_truth_col in df.columns:
                    st.write("True labels (first 10):", df[ground_truth_col].astype(str).head(10).tolist())
        else:
            st.info("Deep model disabled or not available.")

        # ---------------------- Display Results ----------------------
        df_result = df.copy()
        df_result['Batch_Prediction'] = batch_predictions
        df_result['Online_Prediction'] = online_predictions
        if nn_pred_numeric is not None:
            df_result["NN_Prediction"] = nn_pred_numeric
            if inverse_map and all(pid in inverse_map for pid in nn_pred_numeric):
                df_result["NN_Label"] = pd.Series(nn_pred_numeric).map(inverse_map)
            else:
                df_result["NN_Label"] = nn_labels_for_table

        if inverse_map:
            df_result['Batch_Label'] = df_result['Batch_Prediction'].map(inverse_map)
            df_result['Online_Label'] = df_result['Online_Prediction'].map(inverse_map)

        st.subheader("3. Prediction Results")
        st.dataframe(df_result.head(10))

        # ---------------------- Model Performance Evaluation ----------------------
        if ground_truth_col and ground_truth_col in df.columns:
            true_labels = pd.to_numeric(df[ground_truth_col], errors='coerce')
            batch_predictions_series = pd.Series(batch_predictions).astype(int)
            online_predictions_series = pd.Series(online_predictions).astype(int)

            st.subheader("4. Model Performance")
            if nn_pred_numeric is not None:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col4 = st.columns(3)

            batch_acc = accuracy_score(true_labels, batch_predictions_series)
            online_acc = (online_correct / total_samples) if total_samples > 0 else 0.0

            with col1:
                st.metric("Batch Model Accuracy", f"{batch_acc * 100:.2f}%", help="XGBoost on selected+scaled features")
            with col2:
                st.metric("Online Model Accuracy", f"{online_acc * 100:.2f}%", help="River Hoeffding Tree (online)")

            if nn_pred_numeric is not None:
                nn_acc = accuracy_score(true_labels, pd.Series(nn_pred_numeric).astype(int))
                with col3:
                    st.metric("Deep (MLP) Accuracy", f"{nn_acc * 100:.2f}%")

                hybrid_correct = (
                    (batch_predictions_series == true_labels) |
                    (online_predictions_series == true_labels) |
                    (pd.Series(nn_pred_numeric).astype(int) == true_labels)
                ).sum()
                hybrid_acc = hybrid_correct / len(true_labels)
                with col4:
                    st.metric("Hybrid Accuracy", f"{hybrid_acc * 100:.2f}%")
            else:
                hybrid_correct = (
                    (batch_predictions_series == true_labels) |
                    (online_predictions_series == true_labels)
                ).sum()
                hybrid_acc = hybrid_correct / len(true_labels)
                with col4:
                    st.metric("Hybrid Accuracy", f"{hybrid_acc * 100:.2f}%")

        # Download Results
        st.subheader("6. Download Results")
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")

elif uploaded_file:
    st.warning("Model and support files were not loaded correctly. Please check their presence.")

# Sidebar Section
if online_model:
    st.sidebar.subheader("Online Learning Status")
    st.sidebar.write(f"Online model type: {type(online_model).__name__}")
    st.sidebar.write(f"Online metrics: {online_metrics}")

    if drift_detector.drift_detected:
        st.sidebar.warning("Concept drift detected!")

    if st.sidebar.button("Reset Online Model"):
        online_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            ensemble.AdaptiveRandomForestClassifier(n_models=5, seed=42)
        )
        online_metrics = metrics.ClassificationReport()
        joblib.dump(online_model, ONLINE_MODEL_PATH)
        st.sidebar.success("Online model reset!")
