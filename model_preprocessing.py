import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Собираем все train-данные вместе
all_dfs = []
train_files = [f for f in os.listdir("train") if f.endswith(".csv")]

for f in train_files:
    df = pd.read_csv(f"train/{f}")
    all_dfs.append(df)

# Объединяем все данные
combined_df = pd.concat(all_dfs, ignore_index=True)

# Масштабируем на всех данных сразу
scaler = StandardScaler()
combined_df["y_scaled"] = scaler.fit_transform(combined_df[["y"]])

# Возвращаем масштабированные данные обратно в файлы
start_idx = 0
for f in train_files:
    original_df = pd.read_csv(f"train/{f}")
    n_rows = len(original_df)
    end_idx = start_idx + n_rows
    
    original_df["y_scaled"] = combined_df["y_scaled"].iloc[start_idx:end_idx].values
    original_df.to_csv(f"train/{f}", index=False)
    
    start_idx = end_idx

# Сохраняем scaler
joblib.dump(scaler, "scaler.pkl")

print("Preprocessing done.")
