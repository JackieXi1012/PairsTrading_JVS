import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
# Replace yfinance with tiingo
from tiingo import TiingoClient
import warnings

# Configure Tiingo API
# Replace with your actual API key
TIINGO_API_KEY = "fcedaa9129849916e3a2d3fe408f4c0fc8c48952"
# Create client - this will be updated by app.py when an API key is provided
client = TiingoClient({'api_key': TIINGO_API_KEY})

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output


def analyze_pair(stock_a_symbol, stock_b_symbol, initial_cash=1000000, shares_per_trade=100, stop_loss_pct=0.05):
    """
    Analyze a pair of stocks for pairs trading strategy

    Parameters:
    ----------
    stock_a_symbol : str
        Symbol for first stock
    stock_b_symbol : str
        Symbol for second stock
    initial_cash : float
        Initial capital in dollars
    shares_per_trade : int
        Number of shares to trade per signal
    stop_loss_pct : float
        Stop loss percentage (e.g., 0.05 for 5%)

    Returns:
    -------
    dict
        Dictionary containing analysis results, performance metrics, trade records,
        account ledger, and visualization figure
    """
    try:
        # Get the current actual date - always use the current date for end date
        end_date = datetime.datetime.now()

        # For debugging purposes - print current date being used
        print(f"Current date being used as end date: {end_date.strftime('%Y-%m-%d')}")

        # Set date range for 1 year of data
        start_date = end_date - datetime.timedelta(days=365)

        # Format dates for Tiingo API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        print(f"Fetching data for {stock_a_symbol} and {stock_b_symbol}")
        print(f"Date range: {start_date_str} to {end_date_str}")

        # Using Tiingo's API to get historical price data
        try:
            # For stock A - get DataFrame directly with all information
            stock_a_data = client.get_dataframe(
                stock_a_symbol,
                startDate=start_date_str,
                endDate=end_date_str,
                frequency='daily'
            )

            print(
                f"Data fetched for {stock_a_symbol}. Start: {stock_a_data.index.min()}, End: {stock_a_data.index.max()}")

            # For stock B - get DataFrame directly with all information
            stock_b_data = client.get_dataframe(
                stock_b_symbol,
                startDate=start_date_str,
                endDate=end_date_str,
                frequency='daily'
            )

            print(
                f"Data fetched for {stock_b_symbol}. Start: {stock_b_data.index.min()}, End: {stock_b_data.index.max()}")

            # Print the most recent dates to verify data freshness
            print(f"Most recent 5 dates in {stock_a_symbol} data:")
            for date in stock_a_data.index[-5:]:
                print(f"  {date}")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            print("Please check your API key and ticker symbols.")
            return None

        # Ensure both datasets have the same dates by finding common dates
        common_dates = stock_a_data.index.intersection(stock_b_data.index)
        if len(common_dates) < 20:  # Need at least 20 days for meaningful analysis
            print(f"Not enough common trading days between {stock_a_symbol} and {stock_b_symbol}")
            return None

        print(f"Found {len(common_dates)} common trading days")

        # Align data on common dates
        stock_a_data = stock_a_data.loc[common_dates]
        stock_b_data = stock_b_data.loc[common_dates]

        # Create DataFrame with aligned price data
        data = pd.DataFrame({
            'Date': common_dates,
            'Stock_A_Price': stock_a_data['adjClose'].values.flatten(),  # Using adjusted close
            'Stock_B_Price': stock_b_data['adjClose'].values.flatten()
        })

        # Verify we have recent data
        print(f"Final data range: {data['Date'].min()} to {data['Date'].max()}")

        # Ensure Date is datetime type and sort data
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)

        # Calculate log spread and Z-score - EXACTLY as in QMD
        data['Log_Spread'] = np.log(data['Stock_A_Price']) - np.log(data['Stock_B_Price'])
        data['roll_mean'] = data['Log_Spread'].rolling(window=20).mean()
        data['roll_std'] = data['Log_Spread'].rolling(window=20).std()
        data['Z_Score'] = (data['Log_Spread'] - data['roll_mean']) / data['roll_std']

        # Statistical analysis
        # Correlation Analysis - SAME as QMD
        correlation = data['Stock_A_Price'].corr(data['Stock_B_Price'])

        # Cointegration Test - SAME as QMD
        score, pvalue, _ = coint(data['Stock_A_Price'].astype(float), data['Stock_B_Price'].astype(float))

        # ADF Test - SAME as QMD
        adf_result = adfuller(data['Log_Spread'].dropna())

        # Half-life calculation - USING QMD VERSION EXACTLY
        def calculate_half_life(spread):
            """Calculate half-life of the spread series"""
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag

            # Remove missing values
            spread_lag = spread_lag.dropna()
            spread_diff = spread_diff.dropna()

            # Set up regression model: ΔSpread_t = α + ρ*Spread_{t-1} + ε_t
            spread_lag = sm.add_constant(spread_lag)

            # Run OLS regression
            model = sm.OLS(spread_diff.iloc[1:], spread_lag.iloc[1:])
            results = model.fit()

            # Get coefficient for Spread_{t-1} (ρ)
            rho = results.params.iloc[1]

            # Calculate half-life: t_{1/2} = -ln(2)/ln(1+ρ)
            if rho >= 0:  # No mean reversion
                half_life = np.inf
            else:
                half_life = -np.log(2) / rho

            return half_life, results, rho

        half_life, regression_results, rho = calculate_half_life(data['Log_Spread'])

        # Determine maximum holding time based on half-life - SAME as QMD
        max_holding_time = int(1.5 * half_life) if half_life < 100 else 20

        # Try to get VIX data
        try:
            vix_data = client.get_dataframe(
                'VIX',
                startDate=start_date_str,
                endDate=end_date_str,
                frequency='daily'
            )

            # Create a mapping from date string to VIX value for easier lookup
            vix_dict = {d.strftime('%Y-%m-%d'): v for d, v in zip(vix_data.index, vix_data['adjClose'].values)}

            # Create aligned VIX data using the dictionary
            vix_values = []
            for date in data['Date']:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in vix_dict:
                    vix_values.append(vix_dict[date_str])
                else:
                    vix_values.append(None)

            data['VIX'] = vix_values
            # Forward fill manually
            last_valid = 15  # Default starting value
            for i in range(len(data)):
                if pd.isna(data.at[i, 'VIX']):
                    data.at[i, 'VIX'] = last_valid
                else:
                    last_valid = data.at[i, 'VIX']
        except Exception as e:
            print(f"Error getting VIX data: {e}")
            # If no VIX data, use a simple estimate of price volatility - SAME as QMD
            data['VIX'] = data['Stock_A_Price'].pct_change().rolling(20).std() * 100
            data['VIX'].fillna(15, inplace=True)  # Use 15 as default value

        # Adjust thresholds based on VIX - USING QMD APPROACH
        conditions = [
            (data['VIX'] < 20),
            (data['VIX'] >= 20) & (data['VIX'] < 25),
            (data['VIX'] >= 25) & (data['VIX'] < 30),
            (data['VIX'] >= 30)
        ]
        choices = [2.0, 2.25, 2.5, 3.0]
        data['threshold'] = np.select(conditions, choices, default=2.0)

        # Define trading periods - SAME as QMD
        def define_trading_periods(df, period_length_days=5):
            """Divide daily data into trading periods"""
            df = df.copy()
            df = df.sort_values('Date').reset_index(drop=True)
            first_date = df['Date'].iloc[0]
            # Calculate days difference manually
            days_diff = [(date - first_date).days for date in df['Date']]
            df['trading_period'] = [day_diff // period_length_days for day_diff in days_diff]
            return df

        # Set trading periods (each trading period is 5 days)
        period_length_days = 5
        data = define_trading_periods(data, period_length_days)

        # Generate trading signals - SAME as QMD
        data['signal'] = 0
        data.loc[data['Z_Score'] > data['threshold'], 'signal'] = -1  # Short Stock A, Long Stock B
        data.loc[data['Z_Score'] < -data['threshold'], 'signal'] = 1  # Long Stock A, Short Stock B

        # Exit signal (Z-score reverts to ±0.5) - SAME as QMD
        data['exit_signal'] = 0
        data.loc[(data['Z_Score'] <= 0.5) & (data['Z_Score'] >= -0.5), 'exit_signal'] = 1

        # Initialize blotter and ledger
        trading_periods = data['trading_period'].unique()[1:]  # Start from the second trading period

        # Initialize blotter - SAME STRUCTURE as QMD
        blotter = pd.DataFrame({
            'trading_period': trading_periods,
            'entry_timestamp': pd.NaT,
            'qty': 0,
            'exit_timestamp': pd.NaT,
            'entry_price_A': 0.0,
            'entry_price_B': 0.0,
            'exit_price_A': 0.0,
            'exit_price_B': 0.0,
            'success': None,
            'pnl_A': 0.0,
            'pnl_B': 0.0,
            'total_pnl': 0.0,
            'trade_type': None,
            'position_A': None,  # Added for clarity
            'position_B': None  # Added for clarity
        }).set_index('trading_period')

        # Initialize ledger - EXACTLY as QMD
        first_period = data['trading_period'].iloc[0]
        dates_after_first_period = data[data['trading_period'] > first_period]['Date'].tolist()

        ledger = pd.DataFrame({
            'date': dates_after_first_period,
            'position': 0,
            'cash': 0.0,
            'mark_A': data[data['trading_period'] > first_period]['Stock_A_Price'].values,
            'mark_B': data[data['trading_period'] > first_period]['Stock_B_Price'].values,
            'mkt_value': 0.0
        })

        # Set initial cash
        ledger['cash'] = initial_cash

        # Track current position - EXACTLY as QMD
        current_position = 0
        entry_price_A = 0
        entry_price_B = 0
        entry_date = None
        current_period = None

        # Trading execution loop - FOLLOWING QMD LOGIC EXACTLY
        filtered_data = data[data['trading_period'] > data['trading_period'].iloc[0]]

        for i, row in filtered_data.iterrows():
            current_date = row['Date']
            current_period = row['trading_period']

            # Find index for current date in ledger
            ledger_idx_list = ledger[ledger['date'] == current_date].index.tolist()

            if len(ledger_idx_list) == 0:
                continue

            ledger_idx = ledger_idx_list[0]

            # Check if there's an entry signal and no current position
            if row['signal'] != 0 and current_position == 0:
                # Record entry information
                current_position = row[
                                       'signal'] * shares_per_trade  # Positive means long Stock A/short Stock B, negative means short Stock A/long Stock B
                entry_price_A = row['Stock_A_Price']
                entry_price_B = row['Stock_B_Price']
                entry_date = current_date

                # Update blotter
                blotter.loc[current_period, 'entry_timestamp'] = current_date
                blotter.loc[current_period, 'qty'] = abs(current_position)  # Store absolute value for clarity
                blotter.loc[current_period, 'entry_price_A'] = entry_price_A
                blotter.loc[current_period, 'entry_price_B'] = entry_price_B

                if current_position > 0:
                    blotter.loc[current_period, 'trade_type'] = f'Long {stock_a_symbol}, Short {stock_b_symbol}'
                    blotter.loc[current_period, 'position_A'] = 'LONG'
                    blotter.loc[current_period, 'position_B'] = 'SHORT'
                else:
                    blotter.loc[current_period, 'trade_type'] = f'Short {stock_a_symbol}, Long {stock_b_symbol}'
                    blotter.loc[current_period, 'position_A'] = 'SHORT'
                    blotter.loc[current_period, 'position_B'] = 'LONG'

                # Update ledger
                ledger.loc[ledger_idx, 'position'] = current_position

            # Check if there's an exit signal and current position exists - USING QMD LOGIC
            elif ((row['exit_signal'] == 1 or
                   (entry_date and (current_date - entry_date).days > max_holding_time) or
                   (entry_price_A and entry_price_B and
                    abs(((row['Stock_A_Price'] / entry_price_A) - (
                            row['Stock_B_Price'] / entry_price_B)) / 2) > stop_loss_pct))
                  and current_position != 0):

                # Calculate pairs trading P&L
                exit_price_A = row['Stock_A_Price']
                exit_price_B = row['Stock_B_Price']

                # Calculate P&L for both directions - EXACTLY as QMD
                if current_position > 0:  # Long Stock A, Short Stock B
                    pnl_A = current_position * (exit_price_A - entry_price_A)  # Stock A long P&L
                    pnl_B = current_position * (entry_price_B - exit_price_B)  # Stock B short P&L
                else:  # Short Stock A, Long Stock B
                    pnl_A = -current_position * (entry_price_A - exit_price_A)  # Stock A short P&L
                    pnl_B = -current_position * (exit_price_B - entry_price_B)  # Stock B long P&L

                total_pnl = pnl_A + pnl_B

                # Update blotter
                period_of_entry = blotter[blotter['entry_timestamp'] == entry_date].index
                if len(period_of_entry) > 0:
                    period_of_entry = period_of_entry[0]
                    blotter.loc[period_of_entry, 'exit_timestamp'] = current_date
                    blotter.loc[period_of_entry, 'exit_price_A'] = exit_price_A
                    blotter.loc[period_of_entry, 'exit_price_B'] = exit_price_B
                    blotter.loc[period_of_entry, 'success'] = total_pnl > 0
                    blotter.loc[period_of_entry, 'pnl_A'] = pnl_A
                    blotter.loc[period_of_entry, 'pnl_B'] = pnl_B
                    blotter.loc[period_of_entry, 'total_pnl'] = total_pnl

                # Update ledger cash
                if ledger_idx > 0:
                    ledger.loc[ledger_idx, 'cash'] = ledger.loc[ledger_idx - 1, 'cash'] + total_pnl
                else:
                    ledger.loc[ledger_idx, 'cash'] = initial_cash + total_pnl

                # Reset position information
                current_position = 0
                entry_price_A = 0
                entry_price_B = 0
                entry_date = None

            else:
                # If no trade signal, copy cash from previous day
                if ledger_idx > 0:
                    ledger.loc[ledger_idx, 'cash'] = ledger.loc[ledger_idx - 1, 'cash']
                else:
                    ledger.loc[ledger_idx, 'cash'] = initial_cash

            # Update position column in ledger
            ledger.loc[ledger_idx, 'position'] = current_position

            # Calculate market value (considering both positions) - EXACTLY as QMD
            if current_position > 0:  # Long Stock A, Short Stock B
                value_A = current_position * row['Stock_A_Price']  # Stock A long value
                value_B = -current_position * row['Stock_B_Price']  # Stock B short value
            elif current_position < 0:  # Short Stock A, Long Stock B
                value_A = current_position * row['Stock_A_Price']  # Stock A short value
                value_B = -current_position * row['Stock_B_Price']  # Stock B long value
            else:
                value_A = 0
                value_B = 0

            # Total market value = Cash + Stock A position value + Stock B position value
            ledger.loc[ledger_idx, 'mkt_value'] = ledger.loc[ledger_idx, 'cash'] + value_A + value_B

        # Clean blotter, keep only rows with trades
        blotter = blotter[blotter['entry_timestamp'].notna()]

        # Store complete ledger for calculations
        complete_ledger = ledger.copy()

        # Calculate performance metrics
        initial_value = complete_ledger['mkt_value'].iloc[0] if not complete_ledger.empty else initial_cash
        final_value = complete_ledger['mkt_value'].iloc[-1] if not complete_ledger.empty else initial_cash
        total_return = (final_value - initial_value) / initial_value * 100

        # Additional performance metrics - SAME as QMD
        if not complete_ledger.empty:
            complete_ledger['daily_return'] = complete_ledger['mkt_value'].pct_change()
            annualized_return = complete_ledger['daily_return'].mean() * 252 * 100
            annualized_volatility = complete_ledger['daily_return'].std() * np.sqrt(252) * 100
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            max_drawdown = (complete_ledger['mkt_value'] / complete_ledger['mkt_value'].cummax() - 1).min() * 100
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        # Generate summary string
        summary = f"""
- Correlation: {correlation:.4f}
- Cointegration p-value: {pvalue:.4f}
- ADF p-value: {adf_result[1]:.4f}
- Half-life: {half_life:.2f} days
- Maximum holding time: {max_holding_time} days
"""

        # Generate performance summary
        performance = {
            "Initial Capital": f"${initial_cash:,.2f}",
            "Final Value": f"${final_value:,.2f}",
            "Total Return": f"{total_return:.2f}%",
            "Annualized Return": f"{annualized_return:.2f}%",
            "Annualized Volatility": f"{annualized_volatility:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio:.4f}",
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Total Trades": len(blotter)
        }

        # Plot figure - IDENTICAL to QMD
        fig = plt.figure(figsize=(12, 10))

        # Price trend subplot
        plt.subplot(3, 1, 1)
        plt.plot(data['Date'], data['Stock_A_Price'], label=stock_a_symbol)
        plt.plot(data['Date'], data['Stock_B_Price'], label=stock_b_symbol)
        plt.title(f'{stock_a_symbol} vs {stock_b_symbol} Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        # Log spread subplot
        plt.subplot(3, 1, 2)
        plt.plot(data['Date'], data['Log_Spread'])
        plt.title(f'Log Spread (log({stock_a_symbol}) - log({stock_b_symbol}))')
        plt.xlabel('Date')
        plt.ylabel('Log Spread')
        plt.grid(True)

        # Z-score subplot
        plt.subplot(3, 1, 3)
        plt.plot(data['Date'], data['Z_Score'])
        plt.axhline(y=2.0, color='r', linestyle='--', label='Entry Threshold (+2.0)')
        plt.axhline(y=-2.0, color='r', linestyle='--', label='Entry Threshold (-2.0)')
        plt.axhline(y=0.5, color='g', linestyle='--', label='Exit Threshold (+0.5)')
        plt.axhline(y=-0.5, color='g', linestyle='--', label='Exit Threshold (-0.5)')
        plt.title('Z-Score')
        plt.xlabel('Date')
        plt.ylabel('Z-Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        return {
            "summary": summary,
            "performance": performance,
            "blotter": blotter,
            "ledger": ledger,
            "figure": fig
        }

    except Exception as e:
        import traceback
        print(f"Analysis failed: {e}")
        print(traceback.format_exc())
        return None
