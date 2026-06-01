# Install dependencies
import subprocess
subprocess.run(['pip', 'install', 'dagshub', 'mlflow', 'scikit-learn', 'matplotlib', 'seaborn'], check=True)

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# UPLOAD DATASET (khusus Google Colab)
from google.colab import files
print("Upload heart_preprocessing_train.csv dan heart_preprocessing_test.csv")
uploaded = files.upload()

# KONEKSI KE DAGSHUB
os.environ['MLFLOW_TRACKING_USERNAME'] = 'suryani8'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'eb138e8528c3a6a4be25f41f26ee5f8283f65d04'
mlflow.set_tracking_uri('https://dagshub.com/suryani8/Eksperimen_SML_Suryani_apc367d6x0436.mlflow')

# LOAD DATA
# Dataset hasil preprocessing yang sudah siap digunakan untuk training
train_df = pd.read_csv('heart_preprocessing_train.csv')
test_df  = pd.read_csv('heart_preprocessing_test.csv')

# Fix: hapus kolom dataset yang masih ada
train_df = train_df.drop(columns=['dataset'], errors='ignore')
test_df  = test_df.drop(columns=['dataset'], errors='ignore')

# Fix: encode kolom binary yang masih string
le = LabelEncoder()
binary_features = ['sex', 'fbs', 'exang']
for col in binary_features:
    if train_df[col].dtype == 'object':
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test  = test_df.drop(columns=['target'])
y_test  = test_df['target']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# TRAINING DENGAN MLFLOW AUTOLOG
# Menggunakan autolog dari MLflow untuk mencatat semua parameter,
# metrik, dan artefak secara otomatis ke DagsHub
mlflow.set_experiment("Heart Disease - Random Forest")

with mlflow.start_run(run_name="RF_Autolog_Baseline"):

    # Autolog — otomatis log semua parameter dan metrik sklearn
    mlflow.sklearn.autolog()

    # Inisialisasi model Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Melatih model menggunakan data training
    model.fit(X_train, y_train)

    # Prediksi menggunakan data testing
    y_pred = model.predict(X_test)

    # Evaluasi metrik
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n=== Hasil Evaluasi Model ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

print("\nmodelling.py selesai!")
