# =============================================================================
# APP_SIMPLE.PY - Simplified Streamlit Dashboard (No TensorFlow)
# =============================================================================
# This is a simplified version that demonstrates the data loading and
# visualization without the LSTM model. Perfect for testing the setup.
#
# Student: 711524BCS164 | SL.NO: 54
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(
    page_title="AI Stock Price Predictor (Demo)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def download_stock_data(ticker: str = "GOOGL", period: str = "5y"):
    """Download historical stock data from Yahoo Finance."""
    print(f"📥 Downloading data for ticker: {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        
        print(f"[✓] Downloaded {len(df)} rows of data")
        return df
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        raise

def render_header():
    """Render the main header section."""
    st.title("📈 AI Stock Price Predictor (Demo Mode)")
    st.markdown("#### Reg: 711524BCS164 | Demo: Data Loading & Visualization")
    st.markdown("---")

def render_sidebar():
    """Render the sidebar with user controls."""
    st.sidebar.header("⚙️ Configuration")
    
    ticker = st.sidebar.text_input(
        label="Enter Stock Ticker",
        value="GOOGL",
        help="Enter a valid Yahoo Finance ticker (e.g., GOOGL, AAPL, MSFT)"
    )
    
    ticker = ticker.upper().strip()
    run_button = st.sidebar.button(
        label="🚀 Load Data",
        type="primary",
        use_container_width=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📚 **Demo Mode:** This version loads and visualizes stock data. "
        "Full AI prediction requires TensorFlow installation."
    )
    
    return ticker, run_button

def render_data_table(raw_df: pd.DataFrame):
    """Render Finance Dataset Table."""
    st.markdown("### 📊 Finance Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Last 30 Rows of Raw Data:**")
        st.dataframe(
            raw_df.tail(30),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("**Dataset Statistics:**")
        
        total_rows = len(raw_df)
        start_date = raw_df.index[0].strftime('%Y-%m-%d')
        end_date = raw_df.index[-1].strftime('%Y-%m-%d')
        min_price = raw_df['Close'].min()
        max_price = raw_df['Close'].max()
        
        st.metric("Total Rows", f"{total_rows:,}")
        st.metric("Date Range", f"{start_date} to {end_date}")
        st.metric("Min Close Price", f"${min_price:.2f}")
        st.metric("Max Close Price", f"${max_price:.2f}")
        st.metric("Latest Close", f"${raw_df['Close'].iloc[-1]:.2f}")

def render_price_chart(raw_df: pd.DataFrame):
    """Render Price Chart."""
    st.markdown("### 📈 Historical Price Chart")
    
    fig = go.Figure()
    
    # Add Close price line
    fig.add_trace(go.Scatter(
        x=raw_df.index,
        y=raw_df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x}<br>Close: $%{y:.2f}<extra></extra>'
    ))
    
    # Add Volume as secondary axis
    fig.add_trace(go.Scatter(
        x=raw_df.index,
        y=raw_df['Volume'],
        mode='lines',
        name='Volume',
        line=dict(color='lightblue', width=1),
        yaxis='y2',
        opacity=0.3,
        hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Stock Price and Volume History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right'
        ),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_technical_indicators(raw_df: pd.DataFrame):
    """Render simple technical indicators."""
    st.markdown("### 📊 Simple Technical Indicators")
    
    # Calculate moving averages
    df = raw_df.copy()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Create chart with moving averages
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_20'],
        mode='lines',
        name='20-Day MA',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA_50'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        latest_price = df['Close'].iloc[-1]
        latest_ma20 = df['MA_20'].iloc[-1]
        st.metric("Latest Price", f"${latest_price:.2f}")
    
    with col2:
        price_vs_ma20 = ((latest_price - latest_ma20) / latest_ma20) * 100
        st.metric("Price vs 20-MA", f"{price_vs_ma20:+.2f}%")
    
    with col3:
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
        st.metric("Annual Volatility", f"{volatility:.1f}%")

def main():
    """Main application function."""
    render_header()
    ticker, run_button = render_sidebar()
    
    if not run_button:
        st.info(
            "👋 **Welcome to Demo Mode!** Enter a stock ticker in the sidebar "
            "(e.g., GOOGL, AAPL, MSFT) and click **Load Data** to visualize "
            "historical stock prices and technical indicators."
        )
        
        st.markdown("### 📋 What you'll see in this demo:")
        st.markdown("""
        - 📊 **Raw Finance Data**: Last 30 rows of OHLCV data with statistics
        - 📈 **Price Chart**: Interactive chart with price and volume
        - 📊 **Technical Indicators**: Moving averages and volatility metrics
        - ℹ️ **Note**: Full AI prediction requires TensorFlow installation
        """)
        return
    
    try:
        with st.spinner(f"🔄 Loading data for {ticker}..."):
            raw_df = download_stock_data(ticker=ticker, period="5y")
        
        st.success(f"✅ Data loaded successfully for {ticker}!")
        
        # Render all sections
        render_data_table(raw_df)
        st.markdown("---")
        render_price_chart(raw_df)
        st.markdown("---")
        render_technical_indicators(raw_df)
        
        st.markdown("---")
        st.info(
            "🤖 **AI Prediction**: To enable full LSTM-based prediction, "
            "install TensorFlow with: `pip install tensorflow`"
        )
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.error(
            "Please check that the ticker symbol is valid on Yahoo Finance. "
            "Common tickers: GOOGL, AAPL, MSFT, AMZN, TSLA, META"
        )

if __name__ == "__main__":
    main()
