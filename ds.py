import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATA
# =========================
sentiment = pd.read_csv("C:/Users/VIKRANT(N.S)/Downloads/fear_greed_index.csv")
trades = pd.read_csv("C:/Users/VIKRANT(N.S)/Downloads/historical_data.csv")

# =========================
# 2. BASIC CHECKS
# =========================
print("Shapes:", sentiment.shape, trades.shape)
print("\nMissing values (sentiment):\n", sentiment.isna().sum())
print("\nMissing values (trades):\n", trades.isna().sum())

# Remove duplicates
sentiment = sentiment.drop_duplicates()
trades = trades.drop_duplicates()

# =========================
# 3. CLEAN COLUMN NAMES
# =========================
sentiment.columns = sentiment.columns.str.strip().str.lower()
trades.columns = trades.columns.str.strip().str.lower().str.replace(' ', '_')

# =========================
# 4. DATE PROCESSING
# =========================
sentiment['date'] = pd.to_datetime(sentiment['date'])
trades['timestamp_ist'] = pd.to_datetime(trades['timestamp_ist'], dayfirst=True)

# Extract date
sentiment['date'] = sentiment['date'].dt.date
trades['date'] = trades['timestamp_ist'].dt.date

# =========================
# 5. MERGE DATA
# =========================
df = trades.merge(sentiment[['date', 'classification']], on='date', how='left')

# =========================
# 6. FEATURE ENGINEERING
# =========================
df['win'] = df['closed_pnl'] > 0
df['is_long'] = df['side'] == 'BUY'

# =========================
# 7. DAILY METRICS
# =========================
daily = df.groupby(['account', 'date', 'classification']).agg({
    'closed_pnl': 'sum',
    'win': 'mean',
    'size_usd': 'mean',
    'trade_id': 'count',
    'is_long': 'mean'
}).reset_index()

# Rename columns FIRST
daily.rename(columns={
    'closed_pnl': 'daily_pnl',
    'win': 'win_rate',
    'size_usd': 'avg_size',
    'trade_id': 'trades',
    'is_long': 'long_ratio'
}, inplace=True)

# =========================
# 8. ADD RISK METRIC (Drawdown Proxy)
# =========================
daily['drawdown_proxy'] = daily['daily_pnl'].apply(lambda x: min(0, x))

# =========================
# 9. SEGMENTATION
# =========================

# Activity segment
daily['freq_segment'] = daily['trades'].apply(
    lambda x: 'High' if x > daily['trades'].median() else 'Low'
)

# Long/Short bias
daily['bias'] = daily['long_ratio'].apply(
    lambda x: 'Long-heavy' if x > 0.6 else ('Short-heavy' if x < 0.4 else 'Neutral')
)

# Consistency
consistency = daily.groupby('account')['daily_pnl'].std().reset_index()
consistency.columns = ['account', 'pnl_std']

daily = daily.merge(consistency, on='account')

daily['consistency'] = daily['pnl_std'].apply(
    lambda x: 'Consistent' if x < daily['pnl_std'].median() else 'Inconsistent'
)

# =========================
# 10. ANALYSIS
# =========================

# Sentiment comparison
summary = daily.groupby('classification').agg({
    'daily_pnl': 'mean',
    'win_rate': 'mean',
    'trades': 'mean',
    'avg_size': 'mean',
    'long_ratio': 'mean',
    'drawdown_proxy': 'mean'
})

print("\n=== Fear vs Greed Summary ===\n")
print(summary)

# Segment analysis
print("\n=== Segment Analysis (Frequency vs Sentiment) ===\n")
print(daily.groupby(['classification', 'freq_segment'])['daily_pnl'].mean())

# =========================
# 11. VISUALIZATION
# =========================

# PnL distribution
sns.boxplot(x='classification', y='daily_pnl', data=daily)
plt.title("PnL Distribution: Fear vs Greed")
plt.xlabel("Market Sentiment")
plt.ylabel("Daily PnL")
plt.show()

# Trades behavior
sns.boxplot(x='classification', y='trades', data=daily)
plt.title("Trade Frequency by Sentiment")
plt.show()

# =========================
# 12. OPTIONAL MODEL
# =========================
try:
    from sklearn.ensemble import RandomForestClassifier

    daily['profit_label'] = (daily['daily_pnl'] > 0).astype(int)

    X = daily[['trades', 'avg_size', 'long_ratio']]
    y = daily['profit_label']

    model = RandomForestClassifier()
    model.fit(X, y)

    print("\nModel trained successfully!")

except:
    print("\nSkipping model (sklearn not installed)")