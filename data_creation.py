import os
import numpy as np
import pandas as pd

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

def generate_data(anomaly=False, noise=0.0, size=100):
    x = np.linspace(0, 10, size)
    y = 20 + 5 * np.sin(x) + noise * np.random.randn(size)
    if anomaly:
        # добавляем аномалии
        y[20:25] += 30
        y[70:75] -= 30
    return pd.DataFrame({"x": x, "y": y})

for i in range(3):
    df = generate_data(anomaly=(i == 1), noise=2.0, size=100)
    df.to_csv(f"train/data_{i}.csv", index=False)

for i in range(2):
    df = generate_data(anomaly=(i == 1), noise=2.0, size=100)
    df.to_csv(f"test/data_test_{i}.csv", index=False)

print("Data created.")
