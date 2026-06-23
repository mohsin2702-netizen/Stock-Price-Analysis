import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Download stock data (Apple as example)
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
print(df.head())

# Plot closing price trend
plt.figure(figsize=(10,5))
plt.plot(df["Close"], color="steelblue")
plt.title("Apple Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.savefig("stock_plot.png")
plt.close()
print("Plot saved!")

# Feature engineering
df["MA10"] = df["Close"].rolling(10).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["Returns"] = df["Close"].pct_change()
df = df.dropna()

# Features and target
X = df[["Open", "High", "Low", "Volume", "MA10", "MA50", "Returns"]]
y = df["Close"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))

# Save model
joblib.dump(rf, "stock_model.pkl")
print("Model saved!")