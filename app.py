# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import subprocess  # <-- Added
import sys       # <-- Added
import joblib
import os
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from river import compose, preprocessing, linear_model, metrics, drift
from river import tree, ensemble, neighbors, naive_bayes

# NEW: Imports for subprocess
import subprocess
import sys
import time

# NEW: Imports for Confusion Matrix Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO # <-- NEW: For plot download

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
METRICS_PATH = "./model/model_metrics.pkl" # <-- NEW

# NEW: DL artifacts directory (created by your NN trainer)
DL_ARTIFACTS_DIR = "./artifacts"  # must contain nn.pt, scaler.pkl, nn_meta.json

# --- Define the temporary file path for drifted data ---
TEMP_DRIFTED_DATA_PATH = "temp_drifted_data.csv"

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
        # --- FIX: Renamed variable to avoid namespace collision ---
        loaded_metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
        # --- End FIX ---
        
        return model, scaler, selector, reference, label_mapping, online_model, loaded_metrics
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, None, None, None # <-- Updated to 7 Nones

# NEW: cache-load the deep model
@st.cache_resource
def load_nn():
# ... (rest of the function is unchanged) ...
    try:
        return NNPredictor(artifacts_dir=DL_ARTIFACTS_DIR)
    except Exception as e:
        st.warning(f"Deep model not available: {e}")
        return None

# Initialize Models and Metrics
# --- FIX: Updated to receive renamed variable ---
model, scaler, selector, reference_df, label_mapping, online_model, loaded_metrics = load_artifacts()
# ---
# This now correctly refers to the 'river' import
online_metrics = metrics.ClassificationReport()
drift_detector = drift.ADWIN()
inverse_map = {v: k for k, v in label_mapping.items()} if label_mapping else None
nn = load_nn()

# ---------------------- Utils ----------------------

# Drift Detection Function
# ... (function is unchanged) ...
def detect_drift(reference_data, incoming_data, feature_names, threshold=0.05):
# ... (rest of the function is unchanged) ...
    drifted_features = []
    for col in feature_names:
        if col in incoming_data.columns:
            stat, p = ks_2samp(reference_data[col].dropna(), incoming_data[col].dropna())
            if p < threshold:
                drifted_features.append((col, p))
    return drifted_features

# Prepare Data for Online Learning
# ... (function is unchanged) ...
def prepare_river_sample(row, feature_names, target_col=None):
# ... (rest of the function is unchanged) ...
    features = {col: row[col] for col in feature_names if col in row}
    if target_col and target_col in row:
        return features, row[target_col]
    return features

# ---------------------- UI ----------------------

# ... (UI is unchanged) ...
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
            st.warning(f"Drift detected in {len(drifted)} feature(s)! Triggering automated response.")
            
            # --- START: NEW AUTOMATION LOGIC ---
            st.subheader("Automated MLOps Pipeline: Drift Response")

            # --- 1. Save Drifted Data ---
            st.write("Step 1: Saving drifted data for analysis...")
            try:
                # Save the cleaned numeric_df to the temp file
                numeric_df.to_csv(TEMP_DRIFTED_DATA_PATH, index=False) 
                st.write(f"Saved '{TEMP_DRIFTED_DATA_PATH}'.")
            except Exception as e:
                st.error(f"Failed to save drifted data file: {e}")
                st.stop()

            # --- 2. Run Synthetic Data Factory ---
            st.write("Step 2: Running Synthetic Data Factory (Stage 1)...")
            process_1_placeholder = st.empty()
            with st.spinner("Generating new synthetic data... This may take a moment."):
                try:
                    # Pass the temp data path as an argument
                    process_data = subprocess.run(
                        [sys.executable, "synthetic_data_factory.py", TEMP_DRIFTED_DATA_PATH], 
                        capture_output=True, text=True, check=True, timeout=120
                    )
                    process_1_placeholder.text(process_data.stdout) # Show output from script
                    st.success("Synthetic data generation complete.")
                except subprocess.CalledProcessError as e:
                    st.error("Synthetic data generation failed:")
                    st.code(e.stderr)
                    st.stop() # Stop if this fails
                except FileNotFoundError:
                    st.error("Error: `synthetic_data_factory.py` not found. Skipping.")
                    st.stop() # Stop if this fails
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop() # Stop if this fails

            # --- 3. Retrain Model ---
            st.write("Step 3: Retraining batch model (Stage 2)...")
            process_2_placeholder = st.empty()
            with st.spinner("Retraining batch model... This may take a long time."):
                try:
                    # Pass the temp data path as an argument
                    process_train = subprocess.run(
                        [sys.executable, "model_trainer.py", TEMP_DRIFTED_DATA_PATH], 
                        capture_output=True, text=True, check=True, timeout=600 # Increased timeout
                    )
                    process_2_placeholder.text(process_train.stdout) # Show output from script
                    st.success("Model retraining complete.")
                except subprocess.CalledProcessError as e:
                    st.error("Model retraining failed:")
                    st.code(e.stderr)
                    st.stop() # Stop if this fails
                except FileNotFoundError:
                    st.error("Error: `model_trainer.py` not found. Skipping.")
                    st.stop() # Stop if this fails
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop() # Stop if this fails

            # --- 4. Reload Artifacts ---
            st.write("Step 4: Reloading new model artifacts...")
            with st.spinner("Reloading new model artifacts..."):
                st.cache_resource.clear()
                # --- FIX: UPDATED to load renamed variable ---
                model, scaler, selector, reference_df, label_mapping, online_model, loaded_metrics = load_artifacts()
                # ---
                if model is None or scaler is None or selector is None:
                    st.error("Failed to reload artifacts after retraining. Please refresh the page.")
                    st.stop()
                else:
                    st.success("New model artifacts loaded successfully.")
            
            st.subheader("Drift Details")
            for col, p in drifted[:10]:
                st.write(f"- {col} (p = {p:.5f})")
            # ----- END OF NEW LOGIC -----
            
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
            
            plot_accuracies = {} # <-- NEW: To store accuracies for plotting
            
            if nn_pred_numeric is not None:
                col1, col2, col3, col4 = st.columns(4)
            else:
                col1, col2, col4 = st.columns(3)

            batch_acc = accuracy_score(true_labels, batch_predictions_series)
            online_acc = (online_correct / total_samples) if total_samples > 0 else 0.0
            
            plot_accuracies["Batch (XGB)"] = batch_acc
            plot_accuracies["Online (River)"] = online_acc

            with col1:
                st.metric("Batch Model Accuracy", f"{batch_acc * 100:.2f}%", help="XGBoost on selected+scaled features")
            with col2:
                st.metric("Online Model Accuracy", f"{online_acc * 100:.2f}%", help="River Hoeffding Tree (online)")

            if nn_pred_numeric is not None:
                nn_acc = accuracy_score(true_labels, pd.Series(nn_pred_numeric).astype(int))
                plot_accuracies["Deep (MLP)"] = nn_acc # <-- NEW
                with col3:
                    st.metric("Deep (MLP) Accuracy", f"{nn_acc * 100:.2f}%")

                hybrid_correct = (
                    (batch_predictions_series == true_labels) |
                    (online_predictions_series == true_labels) |
                    (pd.Series(nn_pred_numeric).astype(int) == true_labels)
                ).sum()
                hybrid_acc = hybrid_correct / len(true_labels)
                plot_accuracies["Hybrid (Any)"] = hybrid_acc # <-- NEW
                with col4:
                    st.metric("Hybrid Accuracy", f"{hybrid_acc * 100:.2f}%")
            else:
                hybrid_correct = (
                    (batch_predictions_series == true_labels) |
                    (online_predictions_series == true_labels)
                ).sum()
                hybrid_acc = hybrid_correct / len(true_labels)
                plot_accuracies["Hybrid (Any)"] = hybrid_acc # <-- NEW
                with col4:
                    st.metric("Hybrid Accuracy", f"{hybrid_acc * 100:.2f}%")

            # --- START: NEW PLOTTING LOGIC FOR A.2 ---
            st.subheader("Figure A.2: Comparative Model Accuracy")
            
            try:
                # Create DataFrame for plotting
                df_acc = pd.DataFrame(plot_accuracies.items(), columns=["Model", "Accuracy"])
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Model", y="Accuracy", data=df_acc, ax=ax, palette="Blues")
                ax.set_title("Comparative Model Accuracy Plot")
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Model Type")
                ax.set_ylim(0, 1.05) # Set y-axis from 0 to 105%
                
                # Add accuracy labels on top of bars
                for p in ax.patches:
                    ax.annotate(f"{p.get_height() * 100:.2f}%", 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', 
                                xytext=(0, 9), 
                                textcoords='offset points')
                
                st.pyplot(fig)
                
                # Add download button for the plot
                plot_buf = BytesIO()
                fig.savefig(plot_buf, format='png', bbox_inches='tight')
                plt.close(fig) # Close the figure to save memory
                
                st.download_button(
                    label="Download Accuracy Plot (A.2)",
                    data=plot_buf.getvalue(),
                    file_name="A2_accuracy_plot.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Could not generate accuracy plot: {e}")
            # --- END: NEW PLOTTING LOGIC FOR A.2 ---


        # Download Results
        st.subheader("6. Download Results")
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction CSV", csv, file_name="predictions.csv", mime="text/csv")

        # --- 7. NEW: DISPLAY CONFUSION MATRIX ---
        st.subheader("7. Retrained Model Confusion Matrix")
        # --- FIX: Use renamed 'loaded_metrics' variable ---
        if loaded_metrics and "confusion_matrix" in loaded_metrics:
            cm = loaded_metrics["confusion_matrix"]
            
            # Use "auto" for labels as the re-encoded labels might not match
            # the original label_mapping. This is the safest approach.
            labels = "auto" 
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=labels, yticklabels=labels)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix for Retrained Batch Model')
            st.pyplot(fig)
            plt.close(fig) # Close the figure to save memory
            
        else:
            st.info("Confusion Matrix is not yet available. It will be generated and displayed here after the first automated retraining cycle.")
        # --- END NEW ---

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e) # NEW: show full traceback

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

