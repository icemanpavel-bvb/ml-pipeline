import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

X_all = []
y_all = []

train_files = os.listdir("train")
for f in train_files:
    if f.endswith(".csv"):
        df = pd.read_csv(f"train/{f}")
        X_all.append(df[["x"]])  
        y_all.append(df["y_scaled"])  

X = pd.concat(X_all, ignore_index=True)
y = pd.concat(y_all, ignore_index=True)

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved.")
