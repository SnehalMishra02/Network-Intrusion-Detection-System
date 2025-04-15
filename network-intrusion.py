import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

MODEL_PATH = "./model/xgb_model.pkl"
SCALER_PATH = "./model/scaler.pkl"
SELECTOR_PATH = "./model/selector.pkl"
REFERENCE_SAMPLE_PATH = "./model/reference_sample.csv"
LABEL_MAPPING_PATH = "./model/label_mapping.pkl"

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)
        reference = pd.read_csv(REFERENCE_SAMPLE_PATH)
        if os.path.exists(LABEL_MAPPING_PATH):
            label_mapping = joblib.load(LABEL_MAPPING_PATH)
        else:
            label_mapping = None
        return model, scaler, selector, reference, label_mapping
    except:
        return None, None, None, None, None

model, scaler, selector, reference_df, label_mapping = load_artifacts()

inverse_map = None
if label_mapping:
    inverse_map = {v: k for k, v in label_mapping.items()}

def detect_drift(reference_data, incoming_data, feature_names, threshold=0.05):
    drifted_features = []
    for col in feature_names:
        if col in incoming_data.columns:
            stat, p = ks_2samp(reference_data[col].dropna(), incoming_data[col].dropna())
            if p < threshold:
                drifted_features.append((col, p))
    return drifted_features

st.title("Real-Time Network Intrusion Detection System")
st.markdown("Upload a CSV file of network traffic to classify entries and detect data drift.")

uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
ground_truth_col = st.text_input("If ground truth is available, enter the label column name (optional):")

retrain_button = st.button("Retrain Model on Incoming Data")

if uploaded_file and model and scaler and selector and reference_df is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("1. Uploaded Data Sample")
        st.dataframe(df.head())

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_df.fillna(numeric_df.mean(numeric_only=True), inplace=True)

        st.subheader("2. Drift Detection")
        common_cols = list(set(reference_df.columns).intersection(numeric_df.columns))
        reference_clean = reference_df[common_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        incoming_clean = numeric_df[common_cols].copy()

        drifted = detect_drift(reference_clean, incoming_clean, common_cols)

        if drifted:
            st.warning(f"Drift detected in {len(drifted)} feature(s):")
            for col, p in drifted:
                st.write(f"- {col} (p = {p:.5f})")
        else:
            st.success("No significant feature drift detected.")

        if retrain_button:
            st.info("Retraining model on combined data...")
            ref_labeled = reference_df.copy()
            inc_labeled = df.copy()

            if ground_truth_col and ground_truth_col in df.columns:
                full_df = pd.concat([ref_labeled, inc_labeled], ignore_index=True)
                full_df = full_df.apply(pd.to_numeric, errors='coerce')
                full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                full_df.fillna(full_df.mean(numeric_only=True), inplace=True)

                X = full_df.drop(columns=[ground_truth_col])
                y = full_df[ground_truth_col]

                if inverse_map:
                    y = y.map(inverse_map)
                    allowed_classes = set(label_mapping.values())
                    valid_mask = y.isin(allowed_classes)
                    removed = (~valid_mask).sum()
                    if removed > 0:
                        st.warning(f"Removed {removed} entries with unknown class labels not in training set.")
                    X = X[valid_mask]
                    y = y[valid_mask]
                else:
                    valid_classes = list(range(y.nunique()))
                    valid_mask = y.isin(valid_classes)
                    X = X[valid_mask]
                    y = y[valid_mask]

                X_selected = X[selector.get_feature_names_out()]
                X_scaled = scaler.transform(X_selected)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                model.fit(X_train, y_train)

                joblib.dump(model, MODEL_PATH)
                st.success("Model retrained successfully.")
            else:
                st.warning("Ground truth labels required for retraining.")

        selected_df = numeric_df.copy()
        selected_df = selected_df[selector.get_feature_names_out()]
        scaled = scaler.transform(selected_df)

        predictions = model.predict(scaled)

        df_result = df.copy()
        df_result['Prediction'] = predictions
        if label_mapping:
            df_result['Prediction_Label'] = df_result['Prediction'].map(label_mapping)

        st.subheader("3. Prediction Results")
        st.dataframe(df_result.head(10))

        if ground_truth_col and ground_truth_col in df.columns:
            true_labels = df[ground_truth_col]
            if label_mapping:
                true_labels = true_labels.map({v: k for k, v in label_mapping.items()})
            acc = accuracy_score(true_labels, predictions)
            st.subheader("4. Classification Accuracy")
            st.success(f"Accuracy: {acc * 100:.2f}%")
            st.text("Classification Report:")
            st.text(classification_report(true_labels, predictions))

        st.subheader("5. Download Results")
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")

elif uploaded_file:
    st.warning("Model and support files were not loaded correctly. Please check their presence.")