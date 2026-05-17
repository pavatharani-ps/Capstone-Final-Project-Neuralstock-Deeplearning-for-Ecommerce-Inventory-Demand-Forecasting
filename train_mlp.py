import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from mlp_model import MLPModel

# ---------------------------------------------------
# Base Project Directory
# ---------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ---------------------------------------------------
# Dataset Path
# ---------------------------------------------------

data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_data.csv"
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(data_path)

print("Dataset Loaded Successfully")

# ---------------------------------------------------
# Remove NaN and Infinite Values
# ---------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df.dropna(inplace=True)

df.reset_index(
    drop=True,
    inplace=True
)

print("Missing Values Removed")

# ---------------------------------------------------
# Select Numeric Column
# ---------------------------------------------------

numeric_columns = df.select_dtypes(
    include='number'
).columns

sales_column = numeric_columns[0]

print(f"Using '{sales_column}' as target column")

# ---------------------------------------------------
# Scaling
# ---------------------------------------------------

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[[sales_column]]
)

print("Scaling Completed")

# ---------------------------------------------------
# Create Sequences
# ---------------------------------------------------

sequence_length = 10

X = []

y = []

for i in range(
    len(scaled_data) - sequence_length
):

    X.append(
        scaled_data[i:i+sequence_length]
    )

    y.append(
        scaled_data[i+sequence_length]
    )

X = np.array(X)

y = np.array(y)

print("Sequence Generation Completed")

# ---------------------------------------------------
# Train/Test Split
# ---------------------------------------------------

train_size = int(len(X) * 0.8)

X_train = X[:train_size]

X_test = X[train_size:]

y_train = y[:train_size]

y_test = y[train_size:]

print("Train Test Split Completed")

# ---------------------------------------------------
# Convert to Tensor
# ---------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

print("Tensor Conversion Completed")

# ---------------------------------------------------
# MLP Model
# ---------------------------------------------------

input_size = (
    X_train.shape[1] *
    X_train.shape[2]
)

model = MLPModel(input_size)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("MLP Model Initialized")

# ---------------------------------------------------
# Training
# ---------------------------------------------------

epochs = 20

for epoch in range(epochs):

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}, Loss: {loss.item()}"
    )

print("Training Completed Successfully")

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

model.eval()

with torch.no_grad():

    predictions = model(X_test)

predictions = predictions.numpy()

# ---------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------

# MAE
mae = mean_absolute_error(
    y_test,
    predictions
)

# RMSE
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

# R2 Score
r2 = r2_score(
    y_test,
    predictions
)

# MAPE
mape = np.mean(
    np.abs(
        (y_test - predictions) / y_test
    )
) * 100

# ---------------------------------------------------
# Print Results
# ---------------------------------------------------

print("\n========== MLP Evaluation ==========")

print("MAE:", mae)

print("RMSE:", rmse)

print("R2 Score:", r2)

print("MAPE:", mape)

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

models_path = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    models_path,
    exist_ok=True
)

model_path = os.path.join(
    models_path,
    "mlp_model.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\nMLP Model Saved Successfully")