# modelling_tuning.py

# Install dependencies
import subprocess
subprocess.run(['pip', 'install', 'dagshub', 'mlflow', 'scikit-learn', 'matplotlib', 'seaborn'], check=True)

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Konfigurasi MLflow Lokal 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Heart Disease - RF Tuning")

# 2. Load Data 
train_df = pd.read_csv('heart_preprocessing_train.csv')
test_df  = pd.read_csv('heart_preprocessing_test.csv')

train_df = train_df.drop(columns=['dataset'], errors='ignore')
test_df  = test_df.drop(columns=['dataset'], errors='ignore')

# Encode
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

# 3. Tuning
param_grid = {'n_estimators' : [50, 100, 200], 'max_depth' : [None, 5, 10], 'min_samples_split': [2, 5]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

# 4. Manual Logging
with mlflow.start_run(run_name="RF_Manual_Tuning"):
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrik
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": auc(*roc_curve(y_test, y_pred_prob)[:2])
    })

    # Artefak (Confusion Matrix & ROC)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    mlflow.sklearn.log_model(best_model, "random_forest_tuned")
    print("Model tuning tersimpan ke MLflow lokal.")
