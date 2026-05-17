import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

processed_dir = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(processed_dir, exist_ok=True)

raw_data_path = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "ecommerce_inventory_demand.csv"
)

df = pd.read_csv(raw_data_path)

# Show columns
print(df.columns)

# Fill missing values
df.ffill(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encoding only if category column exists
if 'category' in df.columns:

    df = pd.get_dummies(df, columns=['category'])

# Scaling
if 'sales' in df.columns:

    scaler = MinMaxScaler()

    df['sales'] = scaler.fit_transform(df[['sales']])

    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    joblib.dump(scaler, scaler_path)

processed_file_path = os.path.join(
    processed_dir,
    "preprocessed.csv"
)

df.to_csv(processed_file_path, index=False)

print("Preprocessing Completed Successfully")