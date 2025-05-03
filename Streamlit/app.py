import streamlit as st
import pandas as pd
from interactive_pairs import analyze_pair, TIINGO_API_KEY
from tiingo import TiingoClient

# Set page config
st.set_page_config(page_title="Pairs Trading Analyzer", layout="wide")

# Title
st.title(" Pairs Trading Analysis Tool")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# API Key input
api_key = st.sidebar.text_input(
    "Tiingo API Key",
    value=TIINGO_API_KEY if TIINGO_API_KEY != "YOUR_API_KEY_HERE" else "",
    type="password"
)

# Stock selection
stock_list = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "BAC", "MA", "DIS", "ADBE", "CRM", "NFLX", "PYPL",
    "INTC", "VZ", "KO", "PEP", "CMCSA", "ABT", "T", "CSCO", "MRK", "PFE", "TMO",
    "ABBV", "ACN", "NKE", "XOM", "CVX", "LLY", "AVGO", "QCOM", "TXN", "MCD",
    "COST", "NEE", "MS", "BMY", "LIN", "SBUX", "AZO", "ORLY", "AMGN", "MDT",
    "HON", "IBM"
]

# Default to AAPL and MSFT as they have reliable data
stock_a = st.sidebar.selectbox("Select Stock A", stock_list, index=0)  # AAPL
stock_b = st.sidebar.selectbox("Select Stock B", stock_list, index=1)  # MSFT

initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000000, step=100000)
shares_per_trade = st.sidebar.number_input("Shares per Trade", value=100, step=10)
stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", min_value=1, max_value=10, value=5)

# Action button
run_analysis = st.sidebar.button("Run Analysis")

# Output area
if run_analysis:
    if not api_key:
        st.error("Please enter your Tiingo API key in the sidebar.")
    else:
        # Update the API key in the interactive_pairs module
        import interactive_pairs

        interactive_pairs.TIINGO_API_KEY = api_key

        # Create a new TiingoClient instance with the updated API key
        interactive_pairs.client = TiingoClient({'api_key': api_key})

        # Show a spinner while running the analysis
        with st.spinner("Running backtest... please wait"):
            result = analyze_pair(stock_a, stock_b, initial_capital, shares_per_trade, stop_loss_pct / 100)

        if result is None:
            st.error("Something went wrong during analysis. Check your inputs or try again later.")
        else:
            st.success("Analysis completed!")

            # Display results
            st.header(f"{stock_a} vs {stock_b} Statistical Summary")
            st.markdown(result["summary"])

            st.header("Performance Metrics")
            st.table(pd.DataFrame(result["performance"], index=["Value"]).T)

            st.header("Trade Blotter")
            st.dataframe(result["blotter"], use_container_width=True)

            st.header("Ledger")
            st.dataframe(result["ledger"].tail(20), use_container_width=True)

            st.header("Visualizations")
            st.pyplot(result["figure"])
else:
    st.markdown(
        """
        👈 Use the sidebar to choose two stocks and define parameters.

        Then click **Run Analysis** to begin your backtest.

        ## About Pairs Trading

        Pairs trading is a market-neutral trading strategy that matches a long position in one stock with a short position in another stock that has a high correlation. The strategy is based on the concept that the two stocks have a long-term price relationship, and when this relationship temporarily weakens (the spread widens), you can profit by betting that the relationship will eventually revert to its historical norm.

        ### How This Tool Works

        1. **Select two correlated stocks** - Ideally from the same sector
        2. **Set your parameters** - Initial capital, trade size, and stop-loss
        3. **Run the analysis** - The app will test a pairs trading strategy using historical data
        4. **Review results** - See performance metrics, trade history, and visualizations

        The strategy enters trades when the Z-score of the price spread exceeds thresholds and exits when it reverts toward the mean.
        """
    )
