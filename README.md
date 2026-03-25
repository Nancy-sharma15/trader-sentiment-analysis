#  Trader Performance vs Market Sentiment

##  Overview
This project analyzes how **market sentiment (Fear vs Greed)** impacts trader behavior and performance on Hyperliquid.

The goal is to identify patterns that can help design **data-driven trading strategies**.


##  Dataset Description

## 1. Bitcoin Market Sentiment
- Columns: `date`, `classification`
- Records: **2644**
- No missing values

## 2. Historical Trader Data
- Records: **211,224**
- Includes:
  - Account
  - Execution Price
  - Trade Size
  - Side (Buy/Sell)
  - Timestamp
  - Closed PnL

       >  Note: Leverage data was not available in the dataset.

##  Methodology

# Data Preparation
- Removed duplicates
- Standardized column names
- Converted timestamps to datetime
- Extracted daily-level data
- Merged datasets on `date`

# Feature Engineering

The following metrics were created:

- **Daily PnL** (per trader)
- **Win Rate**
- **Average Trade Size (USD)**
- **Trade Frequency**
- **Long/Short Ratio**
- **Drawdown Proxy** (negative PnL as risk measure)

## Segmentation

Traders were segmented into:

- **High vs Low Activity** (based on trade count)
- **Directional Bias** (Long-heavy, Short-heavy, Neutral)
- **Consistency** (based on PnL variability)

##  Key Insights

### 1. Higher Profitability During Fear
- Traders generate **higher PnL during Fear and Extreme Fear**
- Suggests better opportunities during market downturns

## 2. Higher Win Rate in Greed
- Win rate peaks during **Extreme Greed**
- But overall profits are lower → smaller gains per trade

## 3. Overtrading During Fear
- Trade frequency increases significantly during **Extreme Fear**
- Indicates reactive behavior in volatile markets


## 4. Larger Positions in Fear
- Traders take **larger positions during Fear**
- Suggests aggressive dip-buying strategies

##  Strategy Recommendations

# Strategy 1 — Fear-Based Opportunity
- Increase position size during **Fear**
- Focus on high-conviction trades

# Strategy 2 — Greed Risk Control
- Reduce position size during **Greed**
- Avoid overconfidence despite higher win rates

# Strategy 3 — Activity Optimization
- Avoid excessive trading during **Extreme Fear**
- Reduce risk of overtrading

## Visualizations

- PnL distribution across sentiment
- Trade frequency comparison
- Long/Short bias analysis

##  Bonus Work

# Predictive Model
- Random Forest classifier used to predict profitability
- Features used:
  - Trade frequency
  - Average size
  - Long/short ratio

## Interactive Dashboard
- Built using **Streamlit**
- Features:
  - Sentiment filter
  - KPI metrics
  - PnL distribution
  - Behavior analysis

## How to Run

# Install dependencies
pip install pandas matplotlib seaborn streamlit scikit-learn

# Run analysis

python ds.py

# Run dashboard

streamlit run dashboard.py

##  Project Structure

trader-sentiment-analysis/
┣ ds.py
┣ dashboard.py
┣ README.md
┗ outputs/

## Conclusion

- Market sentiment significantly affects trader behavior
- Fear-based markets provide better profit opportunities
- Behavioral adjustments can improve trading performance

# Submission

**Role:** Data Science / Analytics Intern  
**Focus:** Trader Behavior Insights  

 # Final Note

This project demonstrates:
- Data cleaning & preprocessing
- Analytical reasoning
- Insight generation
- Practical strategy design
