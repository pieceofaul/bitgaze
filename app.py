import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ============ Streamlit Page Config ============
st.set_page_config(page_title="BitGaze - BTC Forecast Terminal", layout="wide")

# ============ Custom CSS for Bloomberg Look ============
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        background-color: #0e1117;
        color: white;
        font-family: 'Courier New', monospace;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    .stDataFrame div[role="table"] {
        background-color: #1f2329 !important;
    }

    thead tr th {
        background-color: #1f2329 !important;
        color: #ffffff !important;
    }

    tbody tr td {
        color: #cfcfcf !important;
        font-size: 14px;
    }

    footer, header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ============ App Title ============
st.markdown("## 📡 Bitgaze")
st.markdown("## BTC Price Forecast Terminal")
st.markdown("LSTM-based short-term prediction with buy/sell/tp/sl logic")


# ============ Load Model & Data ============
model = load_model("lstm_model.h5")
#btc_data = yf.download('BTC-USD', start='2023-07-17', end='2026-07-17')[['Close']].dropna()
btc_data = yf.download('BTC-USD', period='90d', interval='1d')[['Close']].dropna()


scaler = MinMaxScaler()
btc_data['Close_scaled'] = scaler.fit_transform(btc_data[['Close']])
scaled_data = btc_data[['Close_scaled']].values

# ============ Prepare Prediction Input ============
time_step = 60
def create_input(data, time_step):
    return data[-time_step:].reshape(1, time_step, 1)

future_days = 30
future_data = create_input(scaled_data, time_step)
future_predictions = []

for _ in range(future_days):
    pred = model.predict(future_data, verbose=0)
    future_predictions.append(pred[0][0])
    future_data = np.append(future_data[:, 1:, :], [[[pred[0][0]]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = btc_data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price (USD)': future_prices.flatten()})

# ============ Insight Summary ============
entry_price = float(btc_data['Close'].iloc[-1])
day_7_price = None
day_30_price = None

if len(future_df) >= 30:
    day_7_price = future_df['Predicted Price (USD)'].iloc[6]
    day_30_price = future_df['Predicted Price (USD)'].iloc[-1]

    change_7 = ((day_7_price - entry_price) / entry_price) * 100
    change_30 = ((day_30_price - entry_price) / entry_price) * 100

    st.markdown("### 📊 Prediction Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("🔑 Entry Price (Today)", f"${entry_price:,.2f}")
    col2.metric("📅 Day 7 Forecast", f"${day_7_price:,.2f}", f"{change_7:+.2f}%")
    col3.metric("📅 Day 30 Forecast", f"${day_30_price:,.2f}", f"{change_30:+.2f}%")

# ============ Buy/Sell/TP/SL Logic ============
take_profit_pct = 0.05
stop_loss_pct = 0.03
future_df['Signal'] = 'Hold'
in_position = False
buy_price = 0

for i in range(len(future_df) - 1):
    today = future_df.iloc[i]
    tomorrow = future_df.iloc[i + 1]

    if not in_position and tomorrow['Predicted Price (USD)'] > today['Predicted Price (USD)'] * 1.002:
        future_df.at[i, 'Signal'] = 'Buy'
        buy_price = today['Predicted Price (USD)']
        in_position = True
    elif in_position:
        current_price = today['Predicted Price (USD)']
        change_pct = (current_price - buy_price) / buy_price
        if change_pct >= take_profit_pct:
            future_df.at[i, 'Signal'] = 'Sell (TP)'
            in_position = False
        elif change_pct <= -stop_loss_pct:
            future_df.at[i, 'Signal'] = 'Sell (SL)'
            in_position = False

if in_position:
    future_df.at[len(future_df) - 1, 'Signal'] = 'Sell (End)'

# ============ Display Table ============
st.markdown("### Prediction Table")
st.dataframe(future_df.style.applymap(
    lambda v: 'color: lime' if v == 'Buy' else 'color: red' if 'Sell' in str(v) else 'color: #cfcfcf',
    subset=['Signal']
), use_container_width=True)

# ============ Plot ============
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

ax.plot(future_df['Date'], future_df['Predicted Price (USD)'], label='Predicted Price', color='cyan')

buy_signals = future_df[future_df['Signal'] == 'Buy']
sell_signals = future_df[future_df['Signal'].str.contains('Sell')]

ax.scatter(buy_signals['Date'], buy_signals['Predicted Price (USD)'], marker='^', color='lime', s=100, label='Buy')
ax.scatter(sell_signals['Date'], sell_signals['Predicted Price (USD)'], marker='v', color='red', s=100, label='Sell')

ax.set_title("Predicted BTC Price with Trading Signals", color='white', fontsize=14)
ax.set_xlabel("Date", color='gray')
ax.set_ylabel("Price (USD)", color='gray')
ax.tick_params(colors='gray')
ax.legend(facecolor='#0e1117')
ax.grid(color='gray', linestyle='--', linewidth=0.2)
plt.xticks(rotation=45)
st.pyplot(fig)

# ============ Download Button ============

csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Prediction as CSV",
    data=csv,
    file_name='btc_prediction.csv',
    mime='text/csv',
    help="Download table above as a CSV file"
)