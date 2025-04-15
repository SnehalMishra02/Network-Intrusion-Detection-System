import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("labeled_sample_20k.csv")

# Create a copy for drifted version
drifted_df = df.copy()

# List of 17 features to apply very high drift
drifted_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
    'Fwd Packets/s', 'Packet Length Mean', 'Packet Length Variance',
    'Init_Win_bytes_forward', 'Fwd Header Length.1'
]

# Apply very high drift
for feature in drifted_features:
    if feature in drifted_df.columns:
        multiplier = np.random.uniform(300.0, 800.0)  # Very strong multiplicative drift
        noise = np.random.normal(loc=100, scale=40, size=drifted_df.shape[0])  # Strong additive noise
        drifted_df[feature] = drifted_df[feature] * multiplier + noise
        drifted_df[feature] = np.power(np.abs(drifted_df[feature]), 1/2.5)  # Non-linear transform for complexity

# Save the drifted dataset
drifted_df.to_csv("VeryHighDrifted_sample_20k.csv", index=False)
