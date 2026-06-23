# Stock Price Analysis & Trend

A data science project to analyze and predict stock price trends using historical data.

## Dataset
- Source: Yahoo Finance via yfinance library (Apple - AAPL)

## What this project does
- Downloads real stock data automatically
- Plots closing price trends
- Engineers features like moving averages and returns
- Trains a Random Forest model to predict closing price
- Saves the best model for future use

## Results
- MAE: 0.7573897818724319
- R2 Score: 0.9988799646970762

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python stock.py`