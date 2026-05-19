NeuralStock - Inventory Demand Forecasting using Deep Learning
Project Overview

NeuralStock is a Deep Learning based Inventory Demand Forecasting System developed using LSTM (Long Short-Term Memory) and MLP (Multi-Layer Perceptron) models.
The project predicts future inventory demand using historical sales data and time-series forecasting techniques.

The system helps businesses:

forecast inventory demand
reduce overstock and understock issues
improve supply chain planning
analyze sales trends

Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming Language    |
| Pandas       | Data Processing         |
| NumPy        | Numerical Operations    |
| Scikit-learn | Data Scaling & Metrics  |
| PyTorch      | Deep Learning Framework |
| Streamlit    | Web Application         |
| TensorBoard  | Training Visualization  |
| Render       | Cloud Deployment        |


Deep Learning Models Used
1. LSTM Model
Specialized for time-series forecasting
Learns historical sequential patterns
Predicts future inventory demand
2. MLP Model
Feed Forward Neural Network
Used for comparison with LSTM performance

Project Structure
NeuralStock/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── data_generator.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── mlp_model.py
│   ├── train.py
│   ├── train_mlp.py
│   ├── inference.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md

Workflow

Data Generation
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
Sequence Generation
       ↓
Model Training
       ↓
Evaluation
       ↓
Prediction
       ↓
Streamlit Deployment

Features
Synthetic inventory dataset generation
Data preprocessing and normalization
Time-series feature engineering
LSTM forecasting model
MLP forecasting model
Forecast accuracy evaluation
Streamlit interactive dashboard
Cloud deployment support

Dataset Features

| Column    | Description            |
| --------- | ---------------------- |
| date      | Sales date             |
| category  | Product category       |
| sales     | Inventory sales        |
| promotion | Promotion availability |

Feature Engineering

The following engineered features are created:
| Feature        | Meaning               |
| -------------- | --------------------- |
| lag_7          | Previous 7-day sales  |
| lag_14         | Previous 14-day sales |
| rolling_mean_7 | 7-day average sales   |
| month          | Month feature         |
| day_of_week    | Weekday feature       |


Evaluation Metrics
| Metric   | Meaning                        |
| -------- | ------------------------------ |
| MAE      | Mean Absolute Error            |
| RMSE     | Root Mean Squared Error        |
| MAPE     | Mean Absolute Percentage Error |
| R² Score | Forecasting Accuracy Score     |


# Run Project Files

# Step 1 - Generate Dataset
python src/data_generator.py

# Step 2 - Preprocess Dataset
python src/preprocess.py

# Step 3 - Feature Engineering
python src/feature_engineering.py

# Step 4 - Train LSTM Model
python src/train.py

# Step 5 - Train MLP Model
python src/train_mlp.py

# Step 6 - Run Inference
python src/inference.py

# Step 7 - Run Streamlit App
streamlit run app.py

Deployment

The project is deployed using Render cloud hosting.

Live Application

NeuralStock Live Demo

Sample Prediction

Predicted Demand: 0.4947


Future Enhancements
Real-time inventory tracking
Advanced forecasting models
Sales visualization dashboard
Multi-product forecasting
Cloud database integration

Conclusion

NeuralStock demonstrates how Deep Learning and Time-Series Forecasting can be applied to inventory demand prediction for improving business decision-making and supply chain optimization.
