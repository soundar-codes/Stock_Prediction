# =============================================================================
# DATA_LOADER.PY - Finance Dataset Module
# =============================================================================
# This module handles downloading, preprocessing, and preparing stock market
# data for the LSTM neural network. It fetches 5 years of historical data
# from Yahoo Finance and creates sliding window sequences for training.
#
# Student: 711524BCS164 | SL.NO: 54
# =============================================================================

import yfinance as yf           # Yahoo Finance library for downloading stock data
import numpy as np              # NumPy for numerical operations and array handling
import pandas as pd             # Pandas for DataFrame data structures
from sklearn.preprocessing import MinMaxScaler  # Scaler to normalize price values
from checklist import ChecklistLogger  # Import our checklist logger to track progress


def download_stock_data(ticker: str = "GOOGL", period: str = "5y"):
    """
    Download historical stock data from Yahoo Finance.
    
    This function uses the yfinance library to fetch daily OHLCV
    (Open, High, Low, Close, Volume) data for the specified stock ticker.
    
    Args:
        ticker (str): Stock ticker symbol (default: "GOOGL" for Alphabet Inc.)
        period (str): Time period to download (default: "5y" for 5 years)
    
    Returns:
        pd.DataFrame: Raw stock data with Date index and OHLCV columns
    
    Raises:
        Exception: If download fails (network issues, invalid ticker, etc.)
    """
    print(f"📥 Downloading data for ticker: {ticker}...")
    
    try:
        # Create a Ticker object for the specified stock symbol
        # yfinance.Ticker provides access to historical market data
        stock = yf.Ticker(ticker)
        
        # Download historical data for the specified period
        # 'period' can be: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        df = stock.history(period=period)
        
        # Verify we actually got data back (empty DataFrame means failure)
        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        
        print(f"[✓] Downloaded {len(df)} rows of data")
        return df
        
    except Exception as e:
        # Catch any errors (network, invalid ticker, API issues)
        # and re-raise with a clear error message
        print(f"❌ Error downloading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame, look_back: int = 60):
    """
    Preprocess raw stock data for LSTM training.
    
    This function performs several critical preprocessing steps:
    1. Extracts only the 'Close' price column (our prediction target)
    2. Scales values to 0-1 range using MinMaxScaler (neural networks prefer small inputs)
    3. Creates sliding window sequences: 60 days of history → predict day 61
    4. Splits data into 80% training / 20% testing sets
    
    Args:
        df (pd.DataFrame): Raw stock data from yfinance
        look_back (int): Number of previous days to use as input (default: 60)
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler, raw_close_prices)
            - X_train: Training input sequences (samples, look_back, 1)
            - y_train: Training target values (samples,)
            - X_test: Testing input sequences (samples, look_back, 1)
            - y_test: Testing target values (samples,)
            - scaler: Fitted MinMaxScaler for inverse transformations
            - raw_close_prices: Original unscaled Close prices (for reference)
    """
    print("🔧 Preprocessing data...")
    
    # === STEP 1: Extract Close Price Column ===
    # We only need 'Close' price for prediction (the price at market close)
    # Close price is the most commonly used metric for stock prediction
    close_prices = df[['Close']].copy()  # Double brackets keep it as DataFrame (2D)
    print(f"   → Extracted Close prices. Shape: {close_prices.shape}")
    
    # === STEP 2: Scale Data with MinMaxScaler ===
    # WHY scale? Neural networks perform better with normalized inputs (0-1 range)
    # Stock prices can range from pennies to thousands, scaling puts everything
    # on the same scale, helping the model learn patterns more effectively
    scaler = MinMaxScaler(feature_range=(0, 1))  # All values will be between 0 and 1
    
    # fit_transform learns the min/max from data AND transforms it
    # reshape(-1, 1) converts to column vector required by scikit-learn
    scaled_data = scaler.fit_transform(close_prices)
    print(f"   → Scaled data to [0,1] range. Min: {scaled_data.min():.4f}, Max: {scaled_data.max():.4f}")
    
    # === STEP 3: Create Sliding Window Sequences ===
    # This is the KEY step for time series prediction
    # LSTM needs sequences as input, not single values
    # For each position i, we take 'look_back' days before it as input (X)
    # and the current day as target output (y)
    
    X, y = [], []  # Empty lists to collect sequences
    
    # Loop through the scaled data starting from look_back position
    # This ensures we always have 60 previous days available
    for i in range(look_back, len(scaled_data)):
        # X: Take 'look_back' days ending at position i (positions i-60 to i-1)
        X.append(scaled_data[i-look_back:i, 0])
        # y: The target is the price at position i (the day we're predicting)
        y.append(scaled_data[i, 0])
    
    # Convert lists to NumPy arrays for TensorFlow compatibility
    X = np.array(X)
    y = np.array(y)
    
    print(f"   → Created {len(X)} sequences (each {look_back} days → 1 prediction)")
    
    # === STEP 4: Reshape X for LSTM Input ===
    # LSTM expects 3D input: (samples, time_steps, features)
    # Currently X is 2D: (samples, time_steps)
    # We add a third dimension of size 1 (we have 1 feature: Close price)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(f"   → Reshaped X to LSTM format: {X.shape}")
    
    # === STEP 5: Split into Train/Test Sets (80/20) ===
    # We use first 80% for training, last 20% for testing
    # This maintains temporal order (no shuffling for time series!)
    train_size = int(len(X) * 0.8)  # 80% for training
    
    X_train = X[:train_size]  # First 80% of sequences
    y_train = y[:train_size]  # Corresponding targets
    X_test = X[train_size:]   # Last 20% of sequences
    y_test = y[train_size:]   # Corresponding targets
    
    print(f"   → Split: Train={len(X_train)} samples, Test={len(X_test)} samples")
    
    # Return everything needed for training and evaluation
    return X_train, y_train, X_test, y_test, scaler, close_prices


def load_and_prepare_data(ticker: str = "GOOGL", look_back: int = 60):
    """
    Main function to load and prepare stock data for the entire pipeline.
    
    This is the PRIMARY ENTRY POINT used by app.py and model.py.
    It coordinates downloading, preprocessing, and checklist tracking.
    
    Args:
        ticker (str): Stock symbol to download (default: "GOOGL")
        look_back (int): Number of days to look back (default: 60)
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler, raw_df, checklist_logger)
            All data needed for model training plus the logger for status tracking
    """
    print("\n" + "=" * 60)
    print("   STOCK DATA LOADER - Phase 1: Data Acquisition")
    print("=" * 60 + "\n")
    
    # Initialize the checklist logger to track our progress
    # This will be passed around so other modules can update it too
    checklist = ChecklistLogger()
    
    try:
        # === STEP 1: Download Data ===
        # Fetch raw OHLCV data from Yahoo Finance
        raw_df = download_stock_data(ticker=ticker, period="5y")
        
        # Mark first checklist item as complete
        checklist.mark_done("Finance Dataset Downloaded & Verified")
        
        # === STEP 2: Preprocess Data ===
        # Extract Close prices, scale, create sequences, split train/test
        X_train, y_train, X_test, y_test, scaler, close_prices = preprocess_data(raw_df, look_back)
        
        # Mark preprocessing steps as complete
        checklist.mark_done("Data Scaled with MinMaxScaler")
        checklist.mark_done("Sliding Window Sequences Created (60-day lookback)")
        
        # Print final confirmation with data shapes
        print("\n[✓] Finance Dataset loaded successfully!")
        print(f"    X_train shape: {X_train.shape}")
        print(f"    y_train shape: {y_train.shape}")
        print(f"    X_test shape:  {X_test.shape}")
        print(f"    y_test shape:  {y_test.shape}")
        print(f"    Date range:    {raw_df.index[0].date()} to {raw_df.index[-1].date()}")
        
        # Return everything including the checklist for status tracking
        return X_train, y_train, X_test, y_test, scaler, close_prices, checklist
        
    except Exception as e:
        # If anything fails, mark the relevant items as failed
        checklist.mark_fail("Finance Dataset Downloaded & Verified")
        checklist.mark_fail("Data Scaled with MinMaxScaler")
        checklist.mark_fail("Sliding Window Sequences Created (60-day lookback)")
        
        # Re-raise the exception so caller knows something went wrong
        print(f"\n❌ Data loading failed: {e}")
        raise


# =============================================================================
# SELF-TEST: Run this module directly to test data loading
# =============================================================================
if __name__ == "__main__":
    """
    When this file is run directly, execute a test download and preprocessing.
    This verifies the data pipeline works without running the full application.
    """
    print("🧪 Testing data_loader.py module...\n")
    
    try:
        # Test with default ticker (GOOGL) and 60-day lookback
        X_train, y_train, X_test, y_test, scaler, raw_df, checklist = load_and_prepare_data(
            ticker="GOOGL", 
            look_back=60
        )
        
        # Display final checklist status
        checklist.print_summary()
        
        print("\n✅ Data loader test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Data loader test failed: {e}")
