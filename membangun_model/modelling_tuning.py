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

# Encode kolom binary
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

# 3. Tuning dengan GridSearchCV
param_grid = {'n_estimators' : [50, 100, 200], 'max_depth' : [None, 5, 10], 'min_samples_split': [2, 5]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

# 4. Manual Logging ke MLflow
with mlflow.start_run(run_name="RF_Manual_Tuning"):
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    # Log Metrik
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": auc(fpr, tpr)
    })

    # Artefak 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # Artefak 2: Feature Importance
    plt.figure(figsize=(8, 6))
    feat_imp = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')

    # Artefak 3: ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    mlflow.log_artifact('roc_curve.png')
    
    # Log Model
    mlflow.sklearn.log_model(best_model, "random_forest_tuned")
    print("Model tuning tersimpan lengkap ke MLflow lokal.")
