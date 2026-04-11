# =============================================================================
# APP.PY - Streamlit Dashboard (Main Entry Point)
# =============================================================================
# This is the web application that provides a visual interface for the
# AI stock prediction system. Built with Streamlit, it allows users to:
# 1. Enter a stock ticker symbol
# 2. View raw stock data
# 3. See interactive charts of actual vs predicted prices
# 4. Monitor training progress and model performance metrics
#
# Student: 711524BCS164 | SL.NO: 54
# =============================================================================

import streamlit as st           # Streamlit web framework for Python
import plotly.graph_objects as go  # Plotly for interactive charts
import numpy as np               # NumPy for numerical operations
import pandas as pd              # Pandas for DataFrame handling

# Import our custom modules
from model import run_full_pipeline  # Runs the complete ML pipeline
from checklist import ChecklistLogger  # Tracks modeling progress


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Set up the Streamlit page title, icon, and layout mode
st.set_page_config(
    page_title="AI Stock Price Predictor",   # Browser tab title
    page_icon="📈",                          # Browser tab icon
    layout="wide",                           # Use full screen width
    initial_sidebar_state="expanded"         # Sidebar starts open
)


def render_header():
    """
    Render the main header section with title and subtitle.
    
    This section displays at the top of the page with:
    - A prominent title with an emoji
    - Student registration information
    - Brief model description
    """
    # === SECTION A: Header ===
    st.title("📈 AI Stock Price Predictor")
    st.markdown("#### Reg: 711524BCS164 | Model: LSTM | Ticker: GOOGL")
    st.markdown("---")  # Horizontal divider line


def render_sidebar():
    """
    Render the sidebar with user controls.
    
    The sidebar contains:
    - Ticker input field for stock symbol
    - Run Prediction button to trigger model training
    
    Returns:
        tuple: (ticker_symbol, run_button_clicked)
    """
    # st.sidebar creates content in the left sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Text input for stock ticker symbol
    # Default value is "GOOGL" but user can change it
    ticker = st.sidebar.text_input(
        label="Enter Stock Ticker",    # Label shown above input
        value="GOOGL",                 # Default value
        help="Enter a valid Yahoo Finance ticker (e.g., GOOGL, AAPL, MSFT)"
    )
    
    # Convert to uppercase (tickers are always uppercase)
    ticker = ticker.upper().strip()
    
    # Button to trigger prediction
    # When clicked, it returns True; otherwise False
    run_button = st.sidebar.button(
        label="🚀 Run Prediction",       # Button text with emoji
        type="primary",                  # Makes button blue/highlighted
        use_container_width=True         # Button fills sidebar width
    )
    
    # Add some helpful info at bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip:** Training takes 2-5 minutes depending on your computer. "
        "Early stopping may complete sooner if the model converges."
    )
    
    return ticker, run_button


def render_data_table(raw_df: pd.DataFrame):
    """
    Render SECTION B: Finance Dataset Table.
    
    Displays:
    - Last 30 rows of raw stock data
    - Basic statistics (total rows, date range, min/max Close price)
    
    Args:
        raw_df (pd.DataFrame): Raw OHLCV data from yfinance
    """
    st.markdown("### 📊 Finance Dataset")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])  # 2:1 ratio (table takes more space)
    
    with col1:
        # Display last 30 rows of the raw data
        # Using st.dataframe provides an interactive, sortable table
        st.markdown("**Last 30 Rows of Raw Data:**")
        st.dataframe(
            raw_df.tail(30),           # Get last 30 rows
            use_container_width=True,    # Full width of column
            height=400                   # Fixed height with scrollbar
        )
    
    with col2:
        # Display basic statistics
        st.markdown("**Dataset Statistics:**")
        
        # Create a metrics box for each statistic
        total_rows = len(raw_df)
        start_date = raw_df.index[0].strftime('%Y-%m-%d')
        end_date = raw_df.index[-1].strftime('%Y-%m-%d')
        min_price = raw_df['Close'].min()
        max_price = raw_df['Close'].max()
        
        # st.metric displays a label, value, and optional delta
        st.metric("Total Rows", f"{total_rows:,}")
        st.metric("Date Range", f"{start_date} to {end_date}")
        st.metric("Min Close Price", f"${min_price:.2f}")
        st.metric("Max Close Price", f"${max_price:.2f}")
        st.metric("Latest Close", f"${raw_df['Close'].iloc[-1]:.2f}")


def render_prediction_chart(actuals, predictions, dates):
    """
    Render SECTION C: Plotly Interactive Chart.
    
    Creates an interactive line chart with:
    - Blue line: Actual Close Prices
    - Red dashed line: Predicted Close Prices
    
    Args:
        actuals (array): Actual stock prices
        predictions (array): Predicted stock prices
        dates (DatetimeIndex): Dates for x-axis
    """
    st.markdown("### 📈 Actual vs Predicted Prices")
    
    # Create a new Plotly figure
    fig = go.Figure()
    
    # === ADD ACTUAL PRICES LINE (Blue Solid) ===
    fig.add_trace(go.Scatter(
        x=dates,                       # X-axis: dates
        y=actuals,                     # Y-axis: actual prices
        mode='lines',                  # Draw as connected lines
        name='Actual Close Price',     # Legend label
        line=dict(color='blue', width=2),  # Blue solid line, 2px thick
        hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
                                       # Custom hover tooltip format
    ))
    
    # === ADD PREDICTED PRICES LINE (Red Dashed) ===
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines',
        name='Predicted Close Price',
        line=dict(color='red', width=2, dash='dash'),  # Red dashed line
        hovertemplate='Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>'
    ))
    
    # === UPDATE LAYOUT FOR BETTER APPEARANCE ===
    fig.update_layout(
        title="Stock Price Prediction Results",  # Chart title
        xaxis_title="Date",                        # X-axis label
        yaxis_title="Price (USD)",                 # Y-axis label
        hovermode="x unified",                     # Show both values on hover
        legend=dict(
            yanchor="top",                         # Legend vertical alignment
            y=0.99,
            xanchor="left",                        # Legend horizontal alignment
            x=0.01
        ),
        template="plotly_white",                   # Clean white background theme
        height=500                                 # Chart height in pixels
    )
    
    # Render the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def render_training_loss_chart(history):
    """
    Render SECTION D: Training Loss Chart.
    
    Shows Training Loss vs Validation Loss across epochs.
    This visualizes how well the model learned without overfitting.
    
    Args:
        history: Keras History object containing loss per epoch
    """
    st.markdown("### 📉 Training Progress (Loss Curves)")
    
    # Extract loss values from history object
    epochs = list(range(1, len(history.history['loss']) + 1))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Training loss line (blue)
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines+markers',          # Lines with dots at each point
        name='Training Loss',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Validation loss line (orange)
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    # Update layout
    fig.update_layout(
        title="Model Training: Loss vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss (Mean Squared Error)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanatory text
    st.info(
        "📚 **How to read this chart:** Training loss should decrease over time. "
        "If validation loss starts increasing while training loss decreases, "
        "the model is overfitting (memorizing instead of learning patterns). "
        "Early stopping prevents this by using the best weights."
    )


def render_checklist_and_metrics(checklist, metrics):
    """
    Render SECTION E: Modeling Checklist Panel and Metrics.
    
    Displays:
    - All 7 checklist items with ✅ or ❌ status
    - RMSE, MAPE, and R² Score in metric boxes
    
    Args:
        checklist (ChecklistLogger): Logger with current status
        metrics (dict): Dictionary containing RMSE, MAPE, R2 values
    """
    st.markdown("### ✅ Modeling Checklist & Performance")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display checklist items
        st.markdown("**Pipeline Status:**")
        
        # Get current status from checklist
        status_list = checklist.get_status()
        
        # Display each item with appropriate emoji
        for item, status in status_list:
            if status == "done":
                st.success(f"✅ {item}")   # Green box for done
            elif status == "fail":
                st.error(f"❌ {item}")     # Red box for failed
            else:
                st.warning(f"⏳ {item}")   # Yellow box for pending
    
    with col2:
        # Display metrics in three side-by-side boxes
        st.markdown("**Evaluation Metrics:**")
        
        # Create 3 columns for metrics
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(
                label="RMSE",
                value=f"${metrics['RMSE']:.2f}",
                help="Root Mean Squared Error: Lower is better (average error in dollars)"
            )
        
        with m2:
            st.metric(
                label="MAPE",
                value=f"{metrics['MAPE']:.2f}%",
                help="Mean Absolute Percentage Error: Lower is better (average % error)"
            )
        
        with m3:
            st.metric(
                label="R² Score",
                value=f"{metrics['R2']:.4f}",
                help="R-Squared: Closer to 1.0 is better (how well model fits the data)"
            )


def main():
    """
    Main application function that coordinates the entire Streamlit app.
    
    This is the entry point when the script runs. It:
    1. Renders the header
    2. Renders the sidebar and gets user input
    3. When "Run Prediction" is clicked, executes the ML pipeline
    4. Displays all results (data table, charts, metrics, checklist)
    """
    # === RENDER STATIC UI ELEMENTS ===
    render_header()
    ticker, run_button = render_sidebar()
    
    # === INITIAL STATE (Before clicking Run) ===
    if not run_button:
        # Show welcome/instruction message
        st.info(
            "👋 **Welcome!** Enter a stock ticker in the sidebar (e.g., GOOGL, AAPL, MSFT) "
            "and click **Run Prediction** to start the AI model training. "
            "This process downloads 5 years of data, builds an LSTM neural network, "
            "trains it for up to 50 epochs, and displays interactive charts of the results."
        )
        
        # Display empty placeholders to show what will appear
        st.markdown("### 📋 What you'll see after running:")
        st.markdown("""
        - 📊 **Raw Finance Data**: Last 30 rows of OHLCV data
        - 📈 **Prediction Chart**: Interactive Plotly chart comparing actual vs predicted prices
        - 📉 **Training Loss Curves**: Visualization of model learning progress
        - ✅ **Modeling Checklist**: 7-step pipeline completion status
        - 🎯 **Performance Metrics**: RMSE, MAPE, and R² scores
        """)
        
        # Stop here - wait for user to click the button
        return
    
    # === RUN PREDICTION (When button is clicked) ===
    try:
        # Create a progress spinner while model trains
        # This shows users that something is happening (can take several minutes)
        with st.spinner(f"🔄 Training LSTM model for {ticker}... This may take 2-5 minutes..."):
            # Run the complete ML pipeline
            # This downloads data, builds model, trains, and evaluates
            results = run_full_pipeline(ticker=ticker, look_back=60)
        
        # Show success message
        st.success(f"✅ Model training completed for {ticker}!")
        
        # === EXTRACT RESULTS ===
        model = results['model']
        predictions = results['predictions']
        actuals = results['actuals']
        history = results['history']
        metrics = results['metrics']
        raw_df = results['raw_df']
        checklist = results['checklist']
        
        # Calculate dates for the test set (for chart x-axis)
        # Test set is last 20% of data
        test_size = len(predictions)
        test_dates = raw_df.index[-test_size:]
        
        # === RENDER ALL SECTIONS ===
        
        # SECTION B: Finance Dataset Table
        render_data_table(raw_df)
        
        st.markdown("---")  # Divider
        
        # SECTION C: Prediction Chart
        render_prediction_chart(actuals, predictions, test_dates)
        
        st.markdown("---")
        
        # SECTION D: Training Loss Chart
        render_training_loss_chart(history)
        
        st.markdown("---")
        
        # SECTION E: Checklist and Metrics
        render_checklist_and_metrics(checklist, metrics)
        
    except Exception as e:
        # If anything goes wrong, display error message
        st.error(f"❌ Error during prediction: {str(e)}")
        st.error(
            "Please check that the ticker symbol is valid on Yahoo Finance. "
            "Common tickers: GOOGL, AAPL, MSFT, AMZN, TSLA, META"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================
# This block runs when the script is executed directly (not imported)
# Streamlit runs this automatically when you do: streamlit run app.py
if __name__ == "__main__":
    main()
