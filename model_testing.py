import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

test_files = os.listdir("test")
for f in test_files:
    if f.endswith(".csv"):
        df = pd.read_csv(f"test/{f}")
        X_test = df[["x"]]
        y_true = df["y"]
        
        y_scaled_true = scaler.transform(y_true.values.reshape(-1, 1)).ravel()
        
        # Предсказываем
        y_pred_scaled = model.predict(X_test)
        
        mse = mean_squared_error(y_scaled_true, y_pred_scaled)
        print(f"{f}: MSE = {mse:.4f}")
