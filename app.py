# AI-Driven Global Water Management System
# Dependencies: pandas, numpy, scikit-learn, keras, matplotlib, seaborn
# Install with: pip install pandas numpy scikit-learn keras matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

def simulate_data():
    """Generates mock water management data."""
    # Correct the 'month' column to have length 1000
    num_rows = 1000
    month_values = list(range(1, 13)) * (num_rows // 12)  # Repeat months for the whole year
    month_values += list(range(1, (num_rows % 12) + 1))  # Add remaining months

    # Create the DataFrame with corrected 'month' column
    return pd.DataFrame({
        'region': ['North', 'South', 'East', 'West'] * (num_rows // 4),
        'month': month_values,
        'population_mil': np.random.uniform(1, 10, num_rows),
        'rainfall_mm': np.random.uniform(20, 400, num_rows),
        'temperature_C': np.random.uniform(15, 45, num_rows),
        'groundwater_level': np.random.uniform(5, 40, num_rows),
        'water_demand_mld': np.random.uniform(100, 1500, num_rows),
        'ph': np.random.uniform(6, 9, num_rows),
        'turbidity': np.random.uniform(1, 10, num_rows),
        'is_safe': np.random.choice([0, 1], num_rows, p=[0.3, 0.7]) # 0 for unsafe, 1 for safe
    })



def predict_water_demand(data):
    """
    Predicts future water demand using an LSTM model.

    Args:
        data (pd.DataFrame): DataFrame containing water management data.

    Returns:
        float: Predicted water demand in MLD.
    """
    print("Training LSTM model for water demand prediction...")
    demand_data = data[['population_mil', 'rainfall_mm', 'temperature_C', 'groundwater_level', 'water_demand_mld']].values # Use .values for numpy array

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(demand_data)

    # Prepare data for LSTM (using a lookback window of 10)
    lookback = 10
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, :-1]) # Features from previous time steps
        y.append(scaled_data[i, -1])           # Target (water demand) at current time step
    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1)) # Output layer for predicting a single value (water demand)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=10, batch_size=16, verbose=0) # Set verbose to 0 for less output during training

    # Predict on the last 'lookback' data points
    # We need to scale the input for prediction using the fitted scaler
    last_10_scaled = scaled_data[-lookback:, :-1].reshape(1, lookback, X.shape[2])
    predicted_scaled = model.predict(last_10_scaled)

    # Inverse transform the prediction to get the actual demand value
    # We need a dummy array with the same number of features as the training data
    dummy_array = np.zeros((1, demand_data.shape[1]))
    dummy_array[0, -1] = predicted_scaled[0][0]
    predicted_demand = scaler.inverse_transform(dummy_array)[0][-1]

    print("\nPredicted Water Demand (MLD):", round(predicted_demand, 2))
    return predicted_demand

def classify_water_quality(data):
    """
    Classifies water quality (safe/unsafe) using a RandomForestClassifier.

    Args:
        data (pd.DataFrame): DataFrame containing water management data.
    """
    print("\nTraining RandomForestClassifier for water quality classification...")
    features = data[['ph', 'turbidity']]
    target = data['is_safe']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nWater Quality Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix: Water Quality")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def optimize_allocation(demand, supply=1000):
    """
    Provides a simple water allocation decision based on demand and supply.

    Args:
        demand (float): Predicted water demand.
        supply (float): Available water supply (default is 1000 MLD).

    Returns:
        str: Allocation decision.
    """
    print("\nOptimizing Water Allocation...")
    if demand > supply:
        return f"Predicted demand ({round(demand, 2)} MLD) exceeds supply ({supply} MLD). Consider reducing usage or exploring additional supply sources."
    else:
        return f"Predicted demand ({round(demand, 2)} MLD) is within supply ({supply} MLD). Demand is met."

if __name__ == "__main__":
    # Simulate data
    data = simulate_data()

    # Predict water demand
    predicted_demand = predict_water_demand(data)

    # Classify water quality
    classify_water_quality(data)

    # Optimize water allocation
    decision = optimize_allocation(predicted_demand)
    print("\nWater Allocation Decision:", decision)