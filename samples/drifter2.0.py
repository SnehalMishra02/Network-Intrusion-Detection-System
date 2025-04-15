import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("labeled_sample_20k.csv")

# Copy for drifted version
drifted_df = df.copy()

# Select more features to drift (modify as per your dataset)
drifted_features = [
    'Fwd Packets/s', 'Fwd IAT Min', 'Total Length of Fwd Packets',
    'Fwd Packet Length Mean', 'Init_Win_bytes_forward', 'Fwd Header Length.1'
]

# Apply stronger and varied drift
for feature in drifted_features:
    if feature in drifted_df.columns:
        multiplier = np.random.uniform(150.0, 300.0)  # Stronger multiplicative drift
        noise = np.random.normal(loc=50, scale=15, size=drifted_df.shape[0])  # Additive noise
        # Apply drift
        drifted_df[feature] = drifted_df[feature] * multiplier + noise
        # Optional: Add non-linear transformation for further complexity
        drifted_df[feature] = np.sqrt(np.abs(drifted_df[feature]))  # Ensures numeric stability

# Save the drifted dataset
drifted_df.to_csv("HighDrifted_sample_20k.csv", index=False)
