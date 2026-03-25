import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
@st.cache_data
def load_data():
    sentiment = pd.read_csv("C:/Users/VIKRANT(N.S)/Downloads/fear_greed_index.csv")
    trades = pd.read_csv("C:/Users/VIKRANT(N.S)/Downloads/historical_data.csv")

    # Clean columns
    sentiment.columns = sentiment.columns.str.strip().str.lower()
    trades.columns = trades.columns.str.strip().str.lower().str.replace(' ', '_')

    # Dates
    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date
    trades['timestamp_ist'] = pd.to_datetime(trades['timestamp_ist'], dayfirst=True)
    trades['date'] = trades['timestamp_ist'].dt.date

    # Merge
    df = trades.merge(sentiment[['date', 'classification']], on='date', how='left')

    # Features
    df['win'] = df['closed_pnl'] > 0
    df['is_long'] = df['side'] == 'BUY'

    # Aggregate
    daily = df.groupby(['account', 'date', 'classification']).agg({
        'closed_pnl': 'sum',
        'win': 'mean',
        'size_usd': 'mean',
        'trade_id': 'count',
        'is_long': 'mean'
    }).reset_index()

    daily.rename(columns={
        'closed_pnl': 'daily_pnl',
        'win': 'win_rate',
        'size_usd': 'avg_size',
        'trade_id': 'trades',
        'is_long': 'long_ratio'
    }, inplace=True)

    return daily

daily = load_data()

# UI
st.title("📊 Trader Behavior vs Market Sentiment")

# Filter
sentiment_filter = st.selectbox(
    "Select Market Sentiment",
    options=daily['classification'].unique()
)

filtered = daily[daily['classification'] == sentiment_filter]

# KPIs
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg PnL", round(filtered['daily_pnl'].mean(), 2))
col2.metric("Win Rate", round(filtered['win_rate'].mean(), 2))
col3.metric("Avg Trades", round(filtered['trades'].mean(), 2))

# PnL Distribution
st.subheader("PnL Distribution")

fig, ax = plt.subplots()
sns.histplot(filtered['daily_pnl'], bins=50, ax=ax)
st.pyplot(fig)

# Trade Behavior
st.subheader("Behavior Analysis")

fig2, ax2 = plt.subplots()
sns.boxplot(x='classification', y='trades', data=daily, ax=ax2)
st.pyplot(fig2)

# Long/Short Bias
st.subheader("Long vs Short Bias")

fig3, ax3 = plt.subplots()
sns.boxplot(x='classification', y='long_ratio', data=daily, ax=ax3)
st.pyplot(fig3)

# Raw Data View
st.subheader("Sample Data")
st.dataframe(filtered.head(50))