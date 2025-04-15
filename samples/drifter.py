import pandas as pd
import numpy as np

df = pd.read_csv("labeled_sample_20k.csv")

# Introduce drift into selected columns
drifted_df = df.copy()
drifted_features = ['Fwd Packets/s', 'Fwd IAT Min']  # Replace with actual feature names

for feature in drifted_features:
    if feature in drifted_df.columns:
        drifted_df[feature] = drifted_df[feature] * np.random.uniform(100.2, 211.5) + np.random.normal(10, 1, size=drifted_df.shape[0])

# Save drifted version
drifted_df.to_csv("drifted_sample_20k123456890.csv", index=False)
