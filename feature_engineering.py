import pandas as pd
import os

# Base project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Processed folder path
processed_dir = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

# Load preprocessed dataset
preprocessed_path = os.path.join(
    processed_dir,
    "preprocessed.csv"
)

df = pd.read_csv(preprocessed_path)

# Show columns
print(df.columns)

# Create lag features only if sales column exists
if 'sales' in df.columns:

    # Lag features
    df['lag_7'] = df['sales'].shift(7)

    df['lag_14'] = df['sales'].shift(14)

    # Rolling mean
    df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()

# Date feature engineering
if 'date' in df.columns:

    df['date'] = pd.to_datetime(df['date'])

    # Month feature
    df['month'] = df['date'].dt.month

    # Day of week feature
    df['day_of_week'] = df['date'].dt.dayofweek

# Remove null rows after feature creation
df.dropna(inplace=True)

# Save final dataset
final_data_path = os.path.join(
    processed_dir,
    "final_data.csv"
)

df.to_csv(final_data_path, index=False)

print("Feature Engineering Completed Successfully")