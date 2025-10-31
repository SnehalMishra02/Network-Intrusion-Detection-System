# model_dl/predict_nn.py
import json, os, joblib, numpy as np, pandas as pd, torch
from model_dl.mlp import TabularMLP
import torch.nn.functional as F

class NNPredictor:
    def __init__(self, artifacts_dir="../artifacts", clip_abs=1e12):
        meta_path = os.path.join(artifacts_dir, "nn_meta.json")
        meta = json.load(open(meta_path))
        self.classes = meta["classes"]
        self.feature_cols = meta["feature_cols"]
        in_dim = meta["input_dim"]

        self.clip_abs = clip_abs
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        self.model  = TabularMLP(in_dim, len(self.classes))
        # We saved a state_dict, so this is safe
        self.model.load_state_dict(torch.load(os.path.join(artifacts_dir, "nn.pt"), map_location="cpu"))
        self.model.eval()

    def _prep(self, df: pd.DataFrame, label_col: str | None):
        X = df.drop(columns=[label_col], errors="ignore") if label_col else df.copy()
        X = X.select_dtypes(include=[np.number]).copy()

        # Sanitize like training
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0.0, inplace=True)
        if self.clip_abs and self.clip_abs > 0:
            X = X.clip(lower=-self.clip_abs, upper=self.clip_abs)

        # Align to training columns
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_cols]  # drop extras, order columns
        return X

    def predict(self, df: pd.DataFrame, label_col: str | None = None):
        X = self._prep(df, label_col)
        Xn = self.scaler.transform(X.values.astype(np.float32))
        with torch.no_grad():
            logits = self.model(torch.from_numpy(Xn))
            probs  = F.softmax(logits, dim=1).numpy()
            y_idx  = probs.argmax(axis=1)
            conf   = probs.max(axis=1)
        y_pred = [self.classes[i] for i in y_idx]
        return y_pred, conf
