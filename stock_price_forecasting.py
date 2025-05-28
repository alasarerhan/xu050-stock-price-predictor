import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_components_plotly

# List of stock symbols
symbols = [
    'AEFES.IS', 'AKBNK.IS', 'ALARK.IS', 'ARCLK.IS', 'ASELS.IS', 'ASTOR.IS', 'BIMAS.IS', 'BRSAN.IS', 'CCOLA.IS',
    'CIMSA.IS', 'DOAS.IS', 'EKGYO.IS', 'ENJSA.IS', 'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS',
    'HALKB.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'MGROS.IS', 'ODAS.IS', 'PETKM.IS', 'PGSUS.IS',
    'SAHOL.IS', 'SISE.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TRKCM.IS',
    'TUPRS.IS', 'VAKBN.IS', 'YKBNK.IS', 'ZOREN.IS', 'DOHOL.IS', 'KRDMD.IS', 'SASA.IS', 'ULKER.IS', 'VESTL.IS',
    'TTKOM.IS', 'TSKB.IS', 'HEKTS.IS', 'MAVI.IS', 'OYAKC.IS']

# Title
st.title("Stock Price Prediction")

# Sidebar for selections
with st.sidebar:
    st.header("XU050")
    st.image('bist50.jpg', width=400)
    symbol = st.selectbox("Stock Symbol", symbols)
    days = st.slider("Number of days to predict", min_value=1, max_value=365, value=30)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        forecast_clicked = st.button("Forecast")

# Main area: Display results when forecast button is clicked
if forecast_clicked:
    with st.spinner("Downloading data and starting forecasting..."):
        # Download data
        data = yf.download(tickers=symbol, period='4y', interval='1d', auto_adjust=True, prepost=True)
        data.reset_index(inplace=True)

        # Create a copy for display with all columns
        display_data = data.copy()

        # Prepare data for Prophet
        data = data[['Date', 'Close']]
        data.columns = ['ds', 'y']

        # Display raw data (tail)
        st.write("### Raw Data")
        st.dataframe(display_data.tail())

        # Fit Prophet model with daily seasonality disabled
        model = Prophet(daily_seasonality=False)
        model.add_seasonality(name='daily', period=1, fourier_order=3)
        model.fit(data)

        # Generate future predictions
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        # Define historical and future forecasts
        last_date = data['ds'].max()
        historical_forecast = forecast[forecast['ds'] <= last_date]
        future_forecast = forecast[forecast['ds'] > last_date]

        # Calculate performance metrics for historical data
        errors = data['y'] - historical_forecast['yhat']
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(errors / data['y'])) * 100
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAPE'],
            'Value': [mse, rmse, mape]
        })

        # Plot actual data
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual', line=dict(color='blue')))
        fig1.update_layout(
            title="Actual Prices",
            xaxis_title="Date",
            yaxis_title="Closing Price",
            width=1600,
            height=600
        )

        # Display actual prices chart
        st.write("### Actual Prices")
        st.plotly_chart(fig1)

        # Display components (trend, weekly, yearly)
        st.write("### Trend and Seasonality Components")
        components_fig = plot_components_plotly(model, forecast)
        st.plotly_chart(components_fig)

        # Plot actual, fitted, forecast, and confidence interval
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=historical_forecast['ds'], y=historical_forecast['yhat'], mode='lines', name='Fitted', line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red')))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], mode='lines', line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(173,216,230,0.2)', name='Confidence Interval'))
        fig2.add_shape(type="line", x0=last_date, y0=0, x1=last_date, y1=1, xref='x', yref='paper', line=dict(color="green", width=2, dash="dash"))
        fig2.add_annotation(x=last_date, y=1, yref="paper", text="Forecast starts", showarrow=True, arrowhead=1, ax=0, ay=-40)
        fig2.update_layout(title="Actual and Forecasted Prices with Confidence Interval", xaxis_title="Date", yaxis_title="Close Price", width=1600, height=600)

        # Display forecast plot
        st.write("### Actual and Forecasted Prices")
        st.plotly_chart(fig2)

        # Display performance metrics
        st.write("### Performance Metrics")
        st.dataframe(metrics_df)
