# Simple test to verify the data loading works
import yfinance as yf
import pandas as pd

print("Testing data loading...")

try:
    # Download GOOGL data
    stock = yf.Ticker("GOOGL")
    df = stock.history(period="5y")
    
    print(f"✅ Successfully loaded {len(df)} rows of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Latest close price: ${df['Close'].iloc[-1]:.2f}")
    print("\nSample data:")
    print(df.tail(5))
    
except Exception as e:
    print(f"❌ Error: {e}")
