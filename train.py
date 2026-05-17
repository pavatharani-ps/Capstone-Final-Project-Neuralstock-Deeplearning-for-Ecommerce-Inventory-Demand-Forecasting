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

from torch.utils.tensorboard import SummaryWriter

from model import LSTMModel

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
# Remove NaN
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

# ---------------------------------------------------
# Select Numeric Column
# ---------------------------------------------------

numeric_columns = df.select_dtypes(
    include='number'
).columns

sales_column = numeric_columns[0]

print(f"Using '{sales_column}'")

# ---------------------------------------------------
# Scaling
# ---------------------------------------------------

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[[sales_column]]
)

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

# ---------------------------------------------------
# Convert To Tensor
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

# ---------------------------------------------------
# TensorBoard
# ---------------------------------------------------

writer = SummaryWriter(
    "runs/neuralstock_experiment"
)

# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = LSTMModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

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

    writer.add_scalar(
        "Training Loss",
        loss.item(),
        epoch
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
# Metrics
# ---------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (y_test - predictions) / y_test
    )
) * 100

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== LSTM Evaluation ==========")

print("MAE:", mae)

print("RMSE:", rmse)

print("MAPE:", mape)

print("R2 Score:", r2)

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
    "lstm_model.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

writer.close()

print("\nLSTM Model Saved Successfully")