# =============================================================================
# AI-Based Stock Price Prediction System
# =============================================================================

**Student:** 711524BCS164 | **SL.NO:** 54  
**Tech Stack:** Python, yfinance, TensorFlow/Keras, Streamlit, Plotly

---

## Project Description

This project implements an AI-powered stock price prediction system using **LSTM (Long Short-Term Memory)** neural networks. It downloads 5 years of historical stock data from Yahoo Finance, preprocesses it using sliding window sequences, trains a deep learning model to learn price patterns, and provides an interactive web dashboard for visualization.

The system predicts the next day's closing stock price based on the previous 60 days of historical prices.

---

## Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

This command installs:
- `yfinance` - Downloads stock market data from Yahoo Finance
- `tensorflow` - Deep learning framework for building LSTM models
- `scikit-learn` - Data preprocessing and evaluation metrics
- `streamlit` - Web application framework for the dashboard
- `plotly` - Interactive charting library
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Additional plotting capabilities

---

## How to Run

Start the Streamlit web application:

```bash
streamlit run app.py
```

This will:
1. Start a local web server
2. Open your browser automatically (or show a URL like `http://localhost:8501`)
3. Display the interactive dashboard

---

## Expected Output

After running the application, you will see:

1. **Header Section**: Title, student registration, and ticker input
2. **Sidebar**: Configuration panel with ticker input and "Run Prediction" button
3. **Data Table**: Last 30 rows of raw OHLCV stock data with statistics
4. **Prediction Chart**: Interactive Plotly chart showing:
   - Blue line: Actual stock prices
   - Red dashed line: AI-predicted prices
5. **Training Loss Chart**: Visualization of model training progress
6. **Checklist Panel**: 7 pipeline steps marked with ✅ or ❌
7. **Metrics Panel**: RMSE, MAPE, and R² scores displayed in metric boxes

The model will train for up to 50 epochs (typically completes in 2-5 minutes with early stopping).

---

## Project Structure

```
stock_prediction/
├── app.py              # Streamlit dashboard (main entry point)
├── data_loader.py      # Downloads & preprocesses stock data
├── model.py            # Builds, trains & evaluates LSTM model
├── checklist.py        # Tracks modeling pipeline progress
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Model Architecture

The LSTM neural network consists of 6 layers:

| Layer | Type | Units | Purpose |
|-------|------|-------|---------|
| 1 | LSTM | 128 | Learns long-range temporal dependencies |
| 2 | Dropout | 0.2 | Prevents overfitting (20% dropout) |
| 3 | LSTM | 64 | Compresses sequence into context vector |
| 4 | Dropout | 0.2 | Additional regularization |
| 5 | Dense | 32 (ReLU) | Non-linear feature combination |
| 6 | Dense | 1 | Final price prediction output |

**Hyperparameters:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Mean Squared Error
- Epochs: 50 (with EarlyStopping patience=5)
- Batch Size: 32
- Validation Split: 10%

---

## Evaluation Metrics

The model is evaluated using three key metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **RMSE** | Root Mean Squared Error | Average error magnitude in dollars (lower is better) |
| **MAPE** | Mean Absolute Percentage Error | Average percentage deviation from actual price (lower is better) |
| **R²** | R-Squared Score | How well model fits data; 1.0 = perfect fit |

---

## Supported Tickers

You can predict any stock available on Yahoo Finance. Common examples:
- `GOOGL` - Alphabet Inc. (default)
- `AAPL` - Apple Inc.
- `MSFT` - Microsoft Corporation
- `AMZN` - Amazon.com Inc.
- `TSLA` - Tesla Inc.
- `META` - Meta Platforms Inc.

---

## Troubleshooting

**Issue:** "No data returned for ticker"
- **Solution:** Verify the ticker symbol is valid on Yahoo Finance

**Issue:** Training takes too long
- **Solution:** The model uses EarlyStopping and typically completes in 2-5 epochs. First run downloads TensorFlow which may take additional time.

**Issue:** Port already in use
- **Solution:** Run with different port: `streamlit run app.py --server.port 8502`

---

## Academic Information

- **Project Title:** AI-Based Stock Price Prediction System
- **Registration Number:** 711524BCS164
- **Serial Number:** 54
- **Technologies Used:** Python, LSTM Neural Networks, Time Series Analysis
- **Data Source:** Yahoo Finance via yfinance API

---

## License

This project is created for academic purposes. Feel free to use and modify for educational use.
