import streamlit as st
import pandas as pd
from interactive_pairs import analyze_pair  # Make sure this matches your file name

# Set page config
st.set_page_config(page_title="Pairs Trading Analyzer", layout="wide")

# Title
st.title("📈 Pairs Trading Analysis Tool")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Stock selection
stock_list = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "BAC", "MA", "DIS", "ADBE", "CRM", "NFLX", "PYPL",
    "INTC", "VZ", "KO", "PEP", "CMCSA", "ABT", "T", "CSCO", "MRK", "PFE", "TMO",
    "ABBV", "ACN", "NKE", "XOM", "CVX", "LLY", "AVGO", "QCOM", "TXN", "MCD",
    "COST", "NEE", "MS", "BMY", "LIN", "SBUX", "AZO", "ORLY", "AMGN", "MDT",
    "HON", "IBM"
]

stock_a = st.sidebar.selectbox("Select Stock A", stock_list, index=48)  # AZO default
stock_b = st.sidebar.selectbox("Select Stock B", stock_list, index=49)  # ORLY default

initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000000, step=100000)
shares_per_trade = st.sidebar.number_input("Shares per Trade", value=100, step=10)
stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", min_value=1, max_value=10, value=5)

# Action button
run_analysis = st.sidebar.button("Run Analysis")

# Output area
if run_analysis:
    st.info("Running backtest... please wait")
    result = analyze_pair(stock_a, stock_b, initial_capital, shares_per_trade, stop_loss_pct / 100)

    if result is None:
        st.error("Something went wrong during analysis. Check your inputs or try again later.")
    else:
        st.success("Analysis completed!")

        # Display results
        st.header(f" {stock_a} vs {stock_b} Statistical Summary")
        st.markdown(result["summary"])

        st.header("Performance Metrics")
        st.table(pd.DataFrame(result["performance"], index=["Value"]).T)

        st.header("Trade Blotter")
        st.dataframe(result["blotter"], use_container_width=True)

        st.header("Ledger")
        st.dataframe(result["ledger"].tail(20), use_container_width=True)

        st.header("📉 Visualizations")
        st.pyplot(result["figure"])
else:
    st.markdown(
        """
        👈 Use the sidebar to choose two stocks and define parameters.

        Then click **"🚀 Run Analysis"** to begin your backtest.
        """
    )
