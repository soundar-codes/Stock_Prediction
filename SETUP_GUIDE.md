# 🚀 Setup Guide - AI Stock Price Prediction System

## ✅ Current Status

**✅ WORKING:** Demo mode with data loading and visualization  
**⚠️ PARTIAL:** Full AI prediction (TensorFlow installation issues on Python 3.13)

---

## 🎯 Quick Start (Working Demo)

1. **Open the demo app:**
   ```bash
   cd "d:\Project AIML\stock_prediction"
   python -m streamlit run app_simple.py --server.port 8502
   ```

2. **Open browser to:** http://localhost:8502

3. **Enter any stock ticker** (GOOGL, AAPL, MSFT, etc.) and click "Load Data"

---

## 📊 What the Demo Shows

- ✅ **Data Loading**: Downloads 5 years of stock data from Yahoo Finance
- ✅ **Interactive Charts**: Price history with volume and moving averages
- ✅ **Technical Indicators**: 20/50-day moving averages, volatility metrics
- ✅ **Statistics**: Date ranges, min/max prices, latest values
- ✅ **Beautiful UI**: Streamlit dashboard with Plotly interactive charts

---

## 🤖 Full AI Prediction (Advanced)

To enable LSTM neural network predictions:

### Option 1: Try TensorFlow Installation
```bash
pip install tensorflow
```

If successful, run:
```bash
python -m streamlit run app.py
```

### Option 2: Use Python 3.11 (Recommended)
TensorFlow has better compatibility with Python 3.11:
```bash
# Create virtual environment with Python 3.11
python -m venv stock_env --python=python3.11
stock_env\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 📁 Project Files

| File | Purpose | Status |
|------|---------|--------|
| `app_simple.py` | **Demo dashboard** (working) | ✅ Ready |
| `app.py` | Full AI dashboard (needs TensorFlow) | ⚠️ Partial |
| `data_loader.py` | Data download & preprocessing | ✅ Ready |
| `model.py` | LSTM neural network | ✅ Ready |
| `checklist.py` | Progress tracking | ✅ Ready |
| `requirements.txt` | Dependencies | ✅ Updated |

---

## 🎓 What You'll Learn

Even in demo mode, this project demonstrates:
- **Data Science**: Working with time series data
- **Web Development**: Streamlit dashboard creation
- **Visualization**: Interactive Plotly charts
- **Finance**: Stock market data and technical indicators
- **API Integration**: Yahoo Finance data fetching

---

## 📱 Demo Features

### 📊 Finance Dataset Table
- Last 30 rows of OHLCV data
- Statistics: total rows, date range, min/max prices

### 📈 Price Chart
- Interactive price and volume chart
- Zoom, pan, hover tooltips
- Beautiful Plotly visualization

### 📊 Technical Indicators
- 20-day and 50-day moving averages
- Price vs moving average comparison
- Annualized volatility calculation

### 🎯 Metrics Dashboard
- Latest price display
- Price relative to moving average
- Volatility percentage

---

## 🔧 Troubleshooting

**"streamlit command not found"**
```bash
python -m streamlit run app_simple.py
```

**"No data returned for ticker"**
- Check ticker symbol (GOOGL, AAPL, MSFT)
- Ensure internet connection
- Try different ticker

**"TensorFlow installation failed"**
- Use demo mode (`app_simple.py`)
- Try Python 3.11 instead of 3.13
- Install Microsoft C++ Build Tools

---

## 🎓 Academic Information

- **Project:** AI-Based Stock Price Prediction System
- **Student:** 711524BCS164 | SL.NO: 54
- **Technologies:** Python, Streamlit, Plotly, yfinance
- **Data Source:** Yahoo Finance API
- **Demo Mode:** Fully functional for data visualization
- **AI Mode:** Requires TensorFlow installation

---

## 🚀 Next Steps

1. **Try the demo** - It's impressive and educational!
2. **Install Python 3.11** for full AI functionality
3. **Study the code** - Rich comments explain every line
4. **Experiment** - Try different stocks and time periods
5. **Learn more** - Read PROJECT_SUMMARY.md for non-technical explanation

---

**Remember:** The demo mode is perfect for learning data visualization, web development, and financial data analysis. The AI prediction is an advanced bonus feature! 🎓
