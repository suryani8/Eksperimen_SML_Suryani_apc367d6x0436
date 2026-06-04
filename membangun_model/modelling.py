import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Konfigurasi MLflow Lokal (WAJIB)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Heart Disease - Random Forest")

# 2. Load & Preprocess Data
train = pd.read_csv('heart_preprocessing_train.csv').drop(columns=['dataset'], errors='ignore')
test = pd.read_csv('heart_preprocessing_test.csv').drop(columns=['dataset'], errors='ignore')

le = LabelEncoder()
for col in ['sex', 'fbs', 'exang']:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X_train, y_train = train.drop(columns=['target']), train['target']
X_test, y_test = test.drop(columns=['target']), test['target']

# 3. Training dengan MLflow Autolog
with mlflow.start_run(run_name="RF_Baseline_Local"):
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Training selesai. Akurasi: {accuracy_score(y_test, y_pred):.4f}")
