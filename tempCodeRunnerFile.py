# Import necessary libraries
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Set environment variables to avoid multithreading issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Load dataset
file_path = r"C:\Users\priya\NIS Project DA\CICIDS2017.csv"  # Change path if needed
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset loaded successfully!")
print(df.info())

# Handling missing values (if any)
df = df.dropna()

# Encoding categorical labels (assuming 'Label' is the target variable)
df['Label'] = df['Label'].astype('category').cat.codes  # Convert attack labels to numeric

# Splitting Features and Target
X = df.drop(columns=['Label'])
y = df['Label']

# Splitting the dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for handling class imbalance
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("SMOTE resampling completed successfully!")

# Feature Scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "trained_model.pkl")
print("Model saved as 'trained_model.pkl'.")

# Visualization of feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()