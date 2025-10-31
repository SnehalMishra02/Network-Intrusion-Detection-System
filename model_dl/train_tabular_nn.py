# model_dl/train_tabular_nn.py
import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from mlp import TabularMLP

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV (e.g., ../samples/labeled_sample_20k.csv)")
    ap.add_argument("--label_col", default="Label", help="Ground-truth column name")
    ap.add_argument("--out", default="../artifacts", help="Artifacts dir (model/scaler/meta)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip_abs", type=float, default=1e12, help="Clip numeric features to ±this value")
    return ap

def main():
    set_seed(42)
    args = make_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    assert args.label_col in df.columns, f"Label column {args.label_col!r} not found."

    # Label
    y_raw = df[args.label_col].astype(str).values

    # Numeric features
    X = df.drop(columns=[args.label_col], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    # Sanitize
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    if args.clip_abs and args.clip_abs > 0:
        X = X.clip(lower=-args.clip_abs, upper=args.clip_abs)

    # Drop zero-variance columns
    non_const_cols = X.columns[X.nunique(dropna=False) > 1]
    X = X[non_const_cols]

    # >>> Save the exact feature column list <<<
    feature_cols = X.columns.tolist()

    # Labels → 0..C-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.astype("float64"))
    X_val = scaler.transform(X_val.astype("float64"))

    # Class weights
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Data
    train_dl = DataLoader(CSVDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(CSVDataset(X_val, y_val), batch_size=2048, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(in_dim=X_tr.shape[1], n_classes=len(classes)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Train (early stop on macro-F1)
    best_f1, best_path = 0.0, os.path.join(args.out, "nn.pt")
    patience, bad = 8, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights.to(device))
            opt.zero_grad(); loss.backward(); opt.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        C = len(classes)
        tp = np.zeros(C, dtype=np.int64)
        fp = np.zeros(C, dtype=np.int64)
        fn = np.zeros(C, dtype=np.int64)
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
                for c in range(C):
                    tp[c] += ((pred == c) & (yb == c)).sum().item()
                    fp[c] += ((pred == c) & (yb != c)).sum().item()
                    fn[c] += ((pred != c) & (yb == c)).sum().item()

        prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)
        rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0)
        f1s  = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec, dtype=float), where=(prec+rec)!=0)
        macro_f1 = float(np.mean(f1s))

        if macro_f1 > best_f1:
            best_f1, bad = macro_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

        print(f"[val] acc={correct/total:.4f} macroF1={macro_f1:.4f}")

    # Save scaler + meta (NOW INCLUDES feature_cols)
    joblib.dump(scaler, os.path.join(args.out, "scaler.pkl"))
    meta = {
        "input_dim": int(X_tr.shape[1]),
        "classes": le.classes_.tolist(),
        "feature_cols": feature_cols
    }
    with open(os.path.join(args.out, "nn_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {best_path}\nDone.")

if __name__ == "__main__":
    main()
