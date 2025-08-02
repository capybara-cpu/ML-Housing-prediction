# regression_model.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def train_and_save_model():
    # Load California housing dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("✅ Model trained successfully")
    print(f"📈 R² Score: {r2:.4f}")
    print(f"📉 Mean Squared Error: {mse:.4f}")

    # Save model to file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved as 'model.pkl'")

if __name__ == "__main__":
    train_and_save_model()
