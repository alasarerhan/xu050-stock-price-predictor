
# XU050 Stock Predictor

This project is a web application built with Streamlit that allows users to predict stock prices for selected Turkish stocks from the XU050 index. It leverages the Prophet library for time series forecasting and Plotly for interactive visualizations.

## Description

The application enables users to select a stock symbol from a predefined list of XU050 stocks and specify the number of days for prediction. It retrieves historical stock data from Yahoo Finance, trains a Prophet model, generates future predictions, and displays the results—including actual prices, forecasts, confidence intervals, and performance metrics.

## Features

- **Stock Selection**: Choose from a list of stock symbols in the XU050 index.
- **Prediction Horizon**: Select the number of days to predict (1 to 365 days) using a slider.
- **Data Download**: Fetches historical stock data from Yahoo Finance for the past 4 years.
- **Model Training**: Trains a Prophet model with daily seasonality on the historical data.
- **Forecasting**: Generates future predictions with confidence intervals.
- **Visualization**: Displays interactive plots of actual prices, fitted values, and forecasts using Plotly.
- **Performance Metrics**: Shows Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) for the historical data.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/xu050-stock-predictor.git
   cd xu050-stock-price-predictor
