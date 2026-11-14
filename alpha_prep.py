# alpha_prep.py


"""
Prepares congressional transaction and market data for model training.
"""

# Standard Libraries
import logging
import re
from pathlib import Path

# Third Party Libraries
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator


# =================================== Data Cleaning/Preparation ========================================================

# Load, prepare, and merge the datasets used by the model
def prepare_data():
    # Load and validate congressional transaction data
    congressional_df = congressional_data()

    # Load and validate VIX market data
    vix_df = vix_data()

    # Create the 10-trading-day VIX target and historical VIX features
    vix_features = prepare_vix_features(vix_df)

    # Align VIX data with each congressional disclosure date
    cleaned_data = merge_datasets(congressional_df, vix_features)

    return cleaned_data


# Load and validate congressional transaction data
def congressional_data():
    # Load raw congressional transaction data.
    df = pd.read_csv(CONGRESSIONAL_DATA_PATH)

    # Convert dates to datetime for validation and later date comparisons.# Convert to datetime for later comparison
    df["transactionDate"] = pd.to_datetime(df["transactionDate"])
    df["disclosureDate"] = pd.to_datetime(df["disclosureDate"])

    # Identify transactions with reported size 'unknown'
    unknown_sizes = (df["transaction_size"] == -0.5).sum()  # A transaction size of -0.5 indicates an unknown size

    # Identify transactions with a transaction date after disclosure
    invalid_date_order = (df["transactionDate"] > df["disclosureDate"]).sum()

    # Count missing transaction or disclosure dates.
    invalid_dates = df[["transactionDate", "disclosureDate"]].isna().sum().sum()

    # Count missing market-data values.
    invalid_values = df[["ticker", "transactionType", "transactionDate", "disclosureDate"]].isna().sum().sum()

    # Log congressional discrepancies
    if any([unknown_sizes > 0, invalid_date_order > 0, invalid_dates > 0, invalid_values > 0]):
        logging.warning(f"Raw congressional data contains %s invalid date order | %s invalid dates"
                        "%s invalid values | %s unknown transaction size",
                        invalid_date_order,
                        invalid_dates,
                        invalid_values,
                        unknown_sizes
                        )

    # Remove data identified data discrepancies
    df = df[df["transaction_size"] != -0.5].copy()  # Remove bad transaction sizes
    df = df[df["transactionDate"] <= df["disclosureDate"]].copy()  # Remove bad dates
    df = df.dropna(subset=["ticker", "transactionType", "transactionDate", "disclosureDate"]).copy()  # Remove bad sizes

    return df


# Load and validate VIX transaction data
def vix_data():
    try:
        # Load raw VIX market data.
        vix_df = pd.read_csv(VIX_PATH)
    except(FileNotFoundError, OSError) as e:
        print(f"Unable to load VIX data: {e}")
        return None


    # Rename source columns to names used throughout the pipeline
    vix_df = vix_df.rename(columns={
        "DATE": "date",
        "CLOSE": "VIX_close"
    })

    # Convert dates to datetime; invalid dates become missing values
    vix_df["date"] = pd.to_datetime(vix_df["date"], errors="coerce")

    # Identify and count records with invalid dates or missing VIX values
    invalid_dates = vix_df["date"].isna().sum()
    invalid_values = vix_df["VIX_close"].isna().sum()
    if invalid_dates > 0 or invalid_values > 0:
        logging.warning(
            "Raw VIX data contains: %s invalid date(s) | %s missing required value(s)",
            invalid_dates,
            invalid_values
        )

    # Remove data discrepancies and sort by date(ascending)
    vix_df = vix_df.dropna(subset=["date"]).copy()
    vix_df = vix_df.sort_values("date").reset_index(drop=True)

    # Log data removal before model fit
    if any([invalid_dates > 0, invalid_values > 0]):
        logging.warning(
            "Discrepancies in VIX data removed before model fit"
        )

    return vix_df


# Create the VIX target and historical features used by the model
def prepare_vix_features(vix_df):
    if vix_df is None:
        return pd.Series({
            "VIX_close": np.nan,
            "VIX_regime_10d": np.nan,
            "VIX_pct_change_10d": np.nan
        })
    # Create the VIX value observed 10 trading days after each VIX observation
    vix_df["future_VIX_10d"] = (vix_df["VIX_close"].shift(-10))

    # Create VIX Target for model predictions
    # Classify the future VIX value as Normal (0) or Elevated (1)
    vix_df["VIX_regime_10d"] = pd.cut(
        vix_df["future_VIX_10d"],
        bins=[-float("inf"), 18.0, float("inf")],  # Normal > VIX value(18) >= Elevated
        labels=False,
        right=False  # Assigns Elevated >= 18
    )

    # Preserve missing target values as pandas nullable integers
    vix_df["VIX_regime_10d"] = (vix_df["VIX_regime_10d"]
                                .astype("Int64")  # Preserve <NA> values
                                )

    # Remove the intermediate target value and unused raw VIX price columns
    vix_df = vix_df.drop(columns=["future_VIX_10d", "OPEN", "HIGH", "LOW"])

    # Create historical 10-trading-day VIX percentage change feature
    vix_df["VIX_pct_change_10d"] = (
            vix_df["VIX_close"] / vix_df["VIX_close"]
            .shift(10) - 1)

    return vix_df


# Align VIX data with congressional transactions by disclosure
def merge_datasets(congressional_df, vix_df):
    vix_columns = [
        "VIX_close",
        "VIX_pct_change_10d",
        "VIX_regime_10d"
    ]

    # Remove existing VIX columns so they can be replaced with the current values from the VIX database
    congressional_df = congressional_df.drop(
        columns=vix_columns,
        errors="ignore"
    )

    # Match each disclosure with the most recent VIX observation
    congressional_df = pd.merge_asof(
        congressional_df.sort_values("disclosureDate"),
        vix_df[
            [
                "date",
                "VIX_close",
                "VIX_pct_change_10d",
                "VIX_regime_10d"
            ]
        ].sort_values("date"),
        left_on="disclosureDate",
        right_on="date",
        direction="backward"  # "backwards" ensures each disclosure uses VIX observations-
    )                         # occurring on or before the disclosure date.

    # Remove VIX date; disclosure date retained
    congressional_df = congressional_df.drop(columns=["date"])

    return congressional_df


# ============================================ User Abilities ==========================================================

# Create a transaction record with engineered features for prediction
def add_transaction(member, ticker, transaction_date, disclosure_date, transaction_type, transaction_range):
    # Load historical data needed to calculate needed features
    congressional_temp = congressional_data()
    vix_history = vix_data()

    # Convert user-provided dates to datetime values
    transaction_date = pd.to_datetime(transaction_date, format="%Y-%m-%d", errors="coerce")
    disclosure_date = pd.to_datetime(disclosure_date, format="%Y-%m-%d", errors="coerce")

    # Reject dates that were not converted; Display errors to user
    if pd.isna(transaction_date) or pd.isna(disclosure_date):
        print("Unable to convert user input dates. Use the format YYYY-MM-DD.")
        return None
    if transaction_date > disclosure_date:  # A transaction cannot occur after it is disclosed
        print(f"Error: Transaction date after disclosure date. Users Dates: T={transaction_date} D={disclosure_date}")
        return None

    # Calculate number of days between disclosure date and transaction date
    disclosure_delay = (disclosure_date - transaction_date).days

    # Check for missing VIX history; Return NaN for missing history
    if vix_history is None:
        logging.info(
            "Unable to generate VIX features for custom transaction, "
            "connect VIX database for more accurate predictions"
        )
        vix_row = pd.Series({
            "VIX_close": np.nan,
            "VIX_regime_10d": np.nan,
            "VIX_pct_change_10d": np.nan
        })
    else:
        # Verify enough existing vix history to calculate VIX features
        vix_history = vix_history[vix_history["date"] <= disclosure_date].tail(120)

        # Assign NaN values to engineered VIX features when not enough history exists
        if vix_history.empty:
            logging.warning(
                "No VIX market data is available on or around disclosure date %s.",
                disclosure_date
            )
            vix_row = pd.Series({
                "VIX_close": np.nan,
                "VIX_regime_10d": np.nan,
                "VIX_pct_change_10d": np.nan
            })
        else:
            # Check that the latest available VIX data is reasonably close to the disclosure date
            latest_vix_date = vix_history["date"].iloc[-1]
            days_since_vix = (disclosure_date - latest_vix_date).days

            if days_since_vix > 3:  # 3 days difference could indicate weekend or holiday
                logging.warning(
                    "Latest VIX data (%s) is too old for disclosure date %s.",
                    latest_vix_date,
                    disclosure_date
                )
                vix_row = pd.Series({
                    "VIX_close": np.nan,
                    "VIX_regime_10d": np.nan,
                    "VIX_pct_change_10d": np.nan
                })
            else:
                vix_history = prepare_vix_features(vix_history)
                vix_row = vix_history.iloc[-1]
                vix_row = vix_row.drop(labels="date")

    # Create base transaction record using user input
    users_row = pd.Series({
        "member.name": member,
        "ticker": ticker,
        "transactionDate": transaction_date,
        "disclosureDate": disclosure_date,
        "disclosureDelay": disclosure_delay,
        "transactionType": transaction_type,
        "transaction_size": calculate_transaction_size(transaction_range),
    })

    # Create event features based on historical congressional activity
    event_features = create_event_features(users_row, congressional_temp)

    # Create technical stock market features
    stock_features = create_stock_features(users_row)

    # Combine the transaction data with all engineered features.
    users_row = pd.concat([
        users_row,
        event_features,
        stock_features,
        vix_row
    ])

    # Convert the resulting series into a single row data frame for predictions
    users_row = users_row.to_frame().T

    return users_row


# =============================== Engineered Congressional Features ====================================================

# Creates features associated with congress persons
def create_event_features(row, df):
    congressional_features = {
        # Transaction
        "transaction_direction": transaction_direction(row["transactionType"]),
        # Associated congressperson activity
        "associated_congressperson_activity_30d": associated_congressperson_activity(row, df, 30),
        "associated_congressperson_activity_60d": associated_congressperson_activity(row, df, 60),
        "associated_congressperson_activity_120d": associated_congressperson_activity(row, df, 120),
        # Transaction size / conviction
        "relative_transaction_size_30d": relative_transaction_size(row, df, 30),
        "relative_transaction_size_60d": relative_transaction_size(row, df, 60),
        "relative_transaction_size_120d": relative_transaction_size(row, df, 120),
        "member_ticker_share_30d": ticker_member_share(row, df, 30),
        "member_ticker_share_60d": ticker_member_share(row, df, 60),
        "member_ticker_share_120d": ticker_member_share(row, df, 120),
        # Congressional market activity
        "congress_net_activity_30d": congressional_net_activity(row, df, 30),
        "congress_net_activity_60d": congressional_net_activity(row, df, 60),
        "congress_net_activity_120d": congressional_net_activity(row, df, 120),
        "congress_transaction_count_30d": congressional_transaction_count(row, df, 30),
        "congress_transaction_count_60d": congressional_transaction_count(row, df, 60),
        "congress_transaction_count_120d": congressional_transaction_count(row, df, 120),
        "unique_congressional_members_30d": ticker_member_breadth(row, df, 30),
        "unique_congressional_members_60d": ticker_member_breadth(row, df, 60),
        "unique_congressional_members_120d": ticker_member_breadth(row, df, 120),
    }

    return pd.Series(congressional_features)


# Parses and finds the mean of an amount range
def calculate_transaction_size(amount_range):
    # Return unknown value indicator(-0.5) when amount range is unknown
    if amount_range == 'unknown':
        return -0.5

    # Transaction amount range parse
    amount_range = str(amount_range).strip()
    amount_range = str(amount_range).strip()

    # Find the two numbers in the input
    amounts = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", amount_range)

    # User input validation
    if len(amounts) != 2:
        print("Amount must contain two values, such as '$1,000-$5,000'.")
        return None

    lower_amount = float(amounts[0].replace(",", ""))
    upper_amount = float(amounts[1].replace(",", ""))

    if lower_amount >= upper_amount:
        print("The lower amount cannot be greater or equal to the upper amount.")
        return None

    # Return the mean of an amount range
    return (upper_amount + lower_amount) / 2


# Assigns a value to a transaction type
def transaction_direction(transaction_type):
    mapping = {
        "Purchase": 1,
        "Sale": -1,
        "Sale (Full)": -1,
        "Sale (Partial)": -0.5
    }
    return mapping.get(transaction_type, 0)


# Count prior trades by the same member in the same stock
def associated_congressperson_activity(row, df, days=60):
    # Details used to define the activity window
    ticker = row["ticker"]
    event_date = row["disclosureDate"]
    congress_person = row["member.name"]
    start_date = event_date - pd.Timedelta(days=days)

    # Find disclosures involving same member and ticker
    associated_activity = df[
        (df["ticker"] == ticker) &
        (df["disclosureDate"] >= start_date) &
        (df["disclosureDate"] < event_date) &
        (df["member.name"] == congress_person)
        ]

    # Return the number of matching prior transactions
    return len(associated_activity)


# Compare the current transaction size with the member's prior average
def relative_transaction_size(row, df, days=60):
    # Details used to define the activity window
    curr_transaction = row["transaction_size"]
    member = row["member.name"]
    event_date = row["disclosureDate"]
    start_date = event_date - pd.Timedelta(days=days)

    # Find member's prior transactions within window
    associated_activity = df[
        (df["member.name"] == member) &
        (df["disclosureDate"] < event_date) &
        (df["disclosureDate"] >= start_date)
        ]

    # If no associated activity still count current transaction
    if len(associated_activity) == 0:
        return 1.0

    # Calculate member's average prior transaction size
    prior_sizes = associated_activity["transaction_size"]
    member_avg = prior_sizes.mean()

    return curr_transaction / member_avg


# Calculate individual member's share of prior congressional activity in a ticker
def ticker_member_share(row, df, days=60):
    # Details used to define the activity window
    ticker = row["ticker"]
    event_date = row["disclosureDate"]
    congress_person = row["member.name"]

    # Create time window
    start_date = event_date - pd.Timedelta(days=days)

    # Find prior congressional transactions matching same ticker
    associated_activity = df[
        (df["ticker"] == ticker) &
        (df["disclosureDate"] >= start_date) &
        (df["disclosureDate"] < event_date)
        ]

    # Return zero when no prior activity exists
    if len(associated_activity) == 0:
        return 0.0

    # Current member's activity within the ticker
    member_activity = associated_activity[
        associated_activity["member.name"] == congress_person
        ]

    # Calculate and return the member's share of all prior activity
    return len(member_activity) / len(associated_activity)


# Calculate net congressional buying and selling activity in a ticker
def congressional_net_activity(row, df, days=60):
    # Details used to define the activity window
    ticker = row["ticker"]
    event_date = (row["disclosureDate"])
    start_date = event_date - pd.Timedelta(days=days)

    # Find congressional transactions within the same ticker
    associated_activity = df[
        (df["ticker"] == ticker) &
        (df["disclosureDate"] < event_date) &
        (df["disclosureDate"] >= start_date)
        ]

    # Assign positive values to purchases and negative values to sales.
    mapping = {
        "Purchase": 1,
        "Sale": -1,
        "Sale (Full)": -1,
        "Sale (Partial)": -1,
        "unknown": -0.5,
    }

    # Sum mapped transactions producing net activity score
    return associated_activity["transactionType"].map(mapping).sum()


# Count prior congressional transactions involving the same ticker
def congressional_transaction_count(row, df, days=60):
    # Details used to define activity window
    ticker = row["ticker"]
    event_date = (row["disclosureDate"])
    start_date = event_date - pd.Timedelta(days=days)

    # Find prior congressional transactions within the ticker
    associated_activity = df[
        (df["ticker"] == ticker) &
        (df["disclosureDate"] < event_date) &
        (df["disclosureDate"] >= start_date)
        ]

    # Return total number of transactions in associated activity
    return len(associated_activity)


# Count distinct congressional members trading the same ticker
def ticker_member_breadth(row, df, days=60):
    # Details used to define activity window
    ticker = row["ticker"]
    event_date = row["disclosureDate"]
    start_date = event_date - pd.Timedelta(days=days)

    # Find prior congressional transactions within the ticker
    associated_activity = df[
        (df["ticker"] == ticker) &
        (df["disclosureDate"] >= start_date) &
        (df["disclosureDate"] < event_date)
        ]
    # Count the number of unique members involved in the activity.
    return associated_activity["member.name"].nunique()


# ========================== Stock Market Statistical Functions ========================================================

# Create technical stock and market features for a transaction
def create_stock_features(row):
    # Load historical price and volume data needed for feature calculations
    history = get_market_history(row, 120)

    # Return expected feature columns NaN if insufficient market history
    if history is None:
        return pd.Series({
            # Returns
            "return_5d": np.nan,
            "return_10d": np.nan,
            "return_20d": np.nan,
            "return_30d": np.nan,
            "return_60d": np.nan,
            "return_120d": np.nan,
            # Volatility
            "volatility_10d": np.nan,
            "volatility_20d": np.nan,
            "volatility_60d": np.nan,
            # Exponential moving average (Trend)
            "ema_distance_20d": np.nan,
            "ema_distance_60d": np.nan,
            # Volume
            "relative_volume_14d": np.nan,
            # Technical indicators
            "ADX_14d": np.nan,
            "ATR_14d": np.nan,
            "MFI_14d": np.nan,
            "RSI": np.nan,
            "Bollinger_position": np.nan,
            "S&P500_return_20d": create_market_feature(row)
        })

    features = {
        # Returns
        "return_5d": calculate_return(row, 5),
        "return_10d": calculate_return(row, 10),
        "return_20d": calculate_return(row, 20),
        "return_30d": calculate_return(row, 30),
        "return_60d": calculate_return(row, 60),
        "return_120d": calculate_return(row, 120),
        # Volatility
        "volatility_10d": calculate_volatility(row, 10),
        "volatility_20d": calculate_volatility(row, 20),
        "volatility_60d": calculate_volatility(row, 60),
        # Exponential moving average (Trend)
        "ema_distance_20d": exponential_moving_average(row, 20),
        "ema_distance_60d": exponential_moving_average(row, 60),
        # Volume
        "relative_volume_14d": relative_volume(row, 14),
        # Technical indicators
        "ADX_14d": average_dir_index(row, 14),
        "ATR_14d": average_true_range(row, 14),
        "MFI_14d": money_flow_index(row, 14),
        "RSI": relative_strength_index(row, 14),
        "Bollinger_position": bollinger_position(row, 20),
        "S&P500_return_20d": create_market_feature(row)

    }

    # Return engineered features as a series
    return pd.Series(features)


# Calculate stock return over a specified number of trading days
def calculate_return(row, return_window):
    # Validate positive return window
    if return_window < 1:
        raise ValueError("return_window must be at least 1")

    # Retrieve historical data and calculate requested return
    history = get_market_history(row, return_window + 1)

    # Return missing value when history is unavailable
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicators
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["close"]].isna().sum().sum()
    validation_log(
        context=f"calculate_return for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records without a closing price
    history = (
        history
            .sort_values("date")
            .dropna(subset=["close"])
    )

    # Validate enough observations for requested window
    if len(history) <= return_window:
        return None

    # Get prices for the return period
    start_price = history["close"].iloc[-(return_window + 1)]
    end_price = history["close"].iloc[-1]

    # Prevent undefined operation (divide by zero)
    if start_price == 0:
        return None

    # Return the percentage of change over the requested period
    return end_price / start_price - 1


# Calculate rolling stock volatility over a window
def calculate_volatility(row, days, annualize=False):
    # Retrieve historical data and sort chronologically
    history = get_market_history(row, days + 1)

    # Return missing value when history is unavailable
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicators
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["close"]].isna().sum().sum()
    validation_log(
        context=f"calculate_volatility for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Drop missing dates
    history = history.sort_values("date").dropna(subset=["close"])

    # Verify enough available observations for window
    if len(history) < days + 1:
        return None

    # Calculate daily percentage returns from the closing prices
    returns = history["close"].pct_change()

    # Calculate standard deviation of returns over the window
    volatility = returns.rolling(days).std().iloc[-1]

    # Return missing value if volatility failed calculation
    if pd.isna(volatility):
        return None
    # Annualized volatility when requested (252 trading days per year)
    if annualize:
        volatility *= 252 ** 0.5

    return volatility


# Calculate how far the current price is from its exponential moving average
def exponential_moving_average(row, days):
    # Retrieve historical data needed to calculate requested EMA and sort chronologically
    history = get_market_history(row, days + 1)

    # Return a missing value when historical data is unavailable.
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["close"]].isna().sum().sum()
    validation_log(
        context=f"EMA for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records without price data
    history = history.sort_values("date").dropna(subset=["close"])

    # Verify enough observations available to calculate EMA
    if len(history) < days:
        return None

    # Get most recent closing price for comparison with EMA
    current_price = history["close"].iloc[-1]

    # Calculate the exponential moving average over the requested window
    moving_average = (
        history["close"]
            .ewm(span=days, adjust=False, min_periods=days)
            .mean()
            .iloc[-1]
    )

    # Return missing value if EMA fails calculation
    if pd.isna(moving_average) or moving_average == 0:
        return None

    # Return current price's relative distance from the EMA
    return (current_price - moving_average) / moving_average


# Measure the strength of the stock's current price trend
def average_dir_index(row, days=14):
    # Retrieve historical data needed to calculate requested average directional index
    history = get_market_history(row, lookback_days=120)

    # Return a missing value when market history is unavailable
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["high", "low", "close"]].isna().sum().sum()
    validation_log(
        context=f"ADR for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records missing required price data.
    history = (
        history
            .sort_values("date")
            .dropna(subset=["high", "low", "close"])
    )

    # Verify enough observations available to calculate ADX
    if len(history) < days + 1:
        return None

    try:
        # Create the ADX indicator using the requested window
        adx_indicator = ADXIndicator(
            high=history["high"],
            low=history["low"],
            close=history["close"],
            window=days,
            fillna=False
        )

        # Calculate the ADX values and use the most recent observation
        adx = adx_indicator.adx()

        # Verify enough observations available to calculate ADX
        if adx.empty:
            return None

        value = adx.iloc[-1]

        # Return missing value if ADX fails calculation
        if pd.isna(value):
            return None

        return value

    # Handle insufficient or invalid data that prevents ADX calculation
    except (IndexError, ValueError):
        return None


# Measure recent price movement relative to the current stock price
def average_true_range(row, days):
    # Retrieve historical data needed to calculate requested average true range
    history = get_market_history(row, days + 1)

    # Return a missing value when historical data is unavailable.
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["high", "low", "close"]].isna().sum().sum()
    validation_log(
        context=f"ATR for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records missing required price data
    history = (
        history
            .sort_values("date")
            .dropna(subset=["high", "low", "close"])
    )

    # Verify enough observations available to calculate ATR
    if len(history) < days:
        return None

    # Calculate the average true range over the requested window
    atr_indicator = AverageTrueRange(
        high=history["high"],
        low=history["low"],
        close=history["close"],
        window=days,
        fillna=False
    )

    # Get the most recent ATR and closing price
    atr = atr_indicator.average_true_range().iloc[-1]
    current_close = history["close"].iloc[-1]

    # Return a missing value if ATR is unavailable or the price is zero
    if pd.isna(atr) or current_close == 0:
        return None

    # Normalize ATR by the current price to make volatility relative to price
    return atr / current_close


# Compare recent trading volume to the stock's normal volume
def relative_volume(row, window):
    # Retrieve historical data needed to calculate relative volume
    history = get_market_history(row, window + 1)

    # Return a missing value when historical data is unavailable.
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history["volume"].isna().sum()
    validation_log(
        context=f"relative_volume for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records without volume data
    history = (history.sort_values("date").dropna(subset=["volume"]))

    # Verify enough observations available to calculate relative volume
    if len(history) < window + 1:
        return None

    # Get most recent trading volume
    current_volume = history["volume"].iloc[-1]

    # Calculate the average volume from the window
    previous_average_volume = (
        history["volume"]
            .shift(1)
            .rolling(window)
            .mean()
            .iloc[-1]
    )

    # Return a missing value when the comparison average is unavailable or zero.
    if pd.isna(previous_average_volume) or previous_average_volume == 0:
        return None

    # Return relative volume
    return current_volume / previous_average_volume


# Measure buying and selling pressure using price and volume
def money_flow_index(row, window):
    # Retrieve historical data needed to calculate the money flow index
    history = get_market_history(row, window + 1)

    # Return a missing value when historical data is unavailable.
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["high", "low", "close", "volume"]].isna().sum().sum()
    validation_log(
        context=f"MFI for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values)

    # Sort chronologically and remove records without price data
    history = (history.sort_values("date").dropna(subset=["high", "low", "close", "volume"]))

    # Verify enough observations available to calculate MFI
    if len(history) < window:
        return None

    # Calculate the MFI over the requested window
    mfi_indicator = MFIIndicator(
        high=history["high"],
        low=history["low"],
        close=history["close"],
        volume=history["volume"],
        window=window,
        fillna=False
    )

    # Get most recent MFI value
    mfi = mfi_indicator.money_flow_index().iloc[-1]

    # Return a missing value when MFI fails calculation
    return None if pd.isna(mfi) else mfi


# Measure the strength of recent upward or downward price movement
def relative_strength_index(row, window=14):
    # Retrieve historical data needed to calculate the relative strength index
    history = get_market_history(row, window + 1)

    # Return a missing value when historical data is unavailable.
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["close"]].isna().sum().sum()
    validation_log(
        context=f"RSI for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records without price data
    history = (history.sort_values("date").dropna(subset=["close"]))

    # Verify enough observations available to calculate RSI
    if len(history) < window:
        return None

    # Calculate the RSI over the requested window
    rsi_indicator = RSIIndicator(
        close=history["close"],
        window=window,
        fillna=False
    )

    # Get most recent RSI value
    rsi = rsi_indicator.rsi().iloc[-1]

    # Return an missing value when RSI fails calculation
    return None if pd.isna(rsi) else rsi


# Measure the stock's current price position within its Bollinger Bands
def bollinger_position(row, window=20, window_dev=2):
    history = get_market_history(row, window + 1)

    # Return a missing value when historical data is unavailable
    if history is None:
        return None

    # Log missing dates or values in the price columns required for the indicator
    invalid_dates = history["date"].isna().sum()
    invalid_values = history[["close"]].isna().sum().sum()
    validation_log(
        context=f"Bollinger position for {row['ticker']} encountered invalid dates: %s  |  invalid values: %s",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values,
    )

    # Sort chronologically and remove records without price data
    history = (history.sort_values("date").dropna(subset=["close"]))

    # Verify enough observations available to calculate Bollinger position
    if len(history) < window:
        return None

    # Calculate the Bollinger position over the requested window
    bollinger = BollingerBands(
        close=history["close"],
        window=window,
        window_dev=window_dev,
        fillna=False
    )

    # Get the current price and most recent upper and lower bands
    current_close = history["close"].iloc[-1]
    upper_band = bollinger.bollinger_hband().iloc[-1]
    lower_band = bollinger.bollinger_lband().iloc[-1]

    # Calculate the total distance between the Bollinger Bands
    band_width = upper_band - lower_band

    # Return a missing value when the bands cannot be compared
    if pd.isna(band_width) or band_width == 0:
        return None

    # Normalize the current price's position within the Bollinger Bands
    return (current_close - lower_band) / band_width

# Calculates the 20-trading-day S&P 500 return relative to a transaction's disclosure date.
# Returns NaN if market data is unavailable or older than 4 days.
def create_market_feature(row):
    try:
        # Load historical S&P 500 data
        sp500_df = pd.read_csv(stooq_file_map.get("SPX"))

        # Rename columns
        sp500_df = sp500_df.rename(
            columns={"<DATE>": "Date", "<CLOSE>": "Close"}
        )

        # 3. Format dates and set 'Date' as the DatetimeIndex
        sp500_df["Date"] = pd.to_datetime(
            sp500_df["Date"].astype(str), format="%Y%m%d"
        )
        sp500_df = sp500_df.sort_values("Date").set_index("Date")

        # 4. Calculate the 20-day return column
        sp500_df["S&P500_return_20d"] = sp500_df["Close"].pct_change(20, fill_method=None)

    # Log loading errors and return NaN if loading fails
    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        pd.errors.ParserError,
        AttributeError,
        ValueError,
        KeyError,
    ) as e:
        logging.warning(
            f"Error connecting to S&P500 data, continuing prediction with NaN values. Exception: {e}"
        )
        return np.nan

    # Verify datetime datatype for error logging
    disclosure_date = pd.to_datetime(row["disclosureDate"])

    # Search against the DatetimeIndex
    idx = sp500_df.index.searchsorted(disclosure_date, side="right") - 1

    # Check if disclosure_date precedes all available S&P 500 data
    if idx < 0:
        logging.warning(
            "No S&P 500 market data available on or before disclosure date %s.",
            disclosure_date.strftime("%Y-%m-%d"),
        )
        return np.nan

    # Extract matched market date
    latest_market_date = sp500_df.index[idx]
    days_since_market = (disclosure_date - latest_market_date).days

    # Check for stale data (> 4 days accommodates 4-day holiday weekends)
    if days_since_market > 4:
        logging.warning(
            "Latest S&P 500 data (%s) is %d days old for disclosure date %s.",
            latest_market_date.strftime("%Y-%m-%d"),
            days_since_market,
            disclosure_date.strftime("%Y-%m-%d"),
        )
        return np.nan

    # Return scalar return value
    return sp500_df["S&P500_return_20d"].iloc[idx]


# ========================== Stooq Database Cleaning and connection ====================================================

# Build a ticker file map for the local stooq market data
def connect_stooq_db():
    STOOQ_TICKER_ALIASES = {
        # ============================================================
        # HISTORICAL / SUCCESSOR TICKERS
        # ============================================================

        "AABA": "GOOG",  # Altaba / Yahoo successor situation
        "ABMD": "JNJ",  # Abiomed -> Johnson & Johnson
        "ACAS": "ARCC",  # American Capital -> Ares Capital
        "ACE": "CB",  # ACE Limited -> Chubb
        "AGN": "ABBV",  # Allergan -> AbbVie
        "ALTR": "INTC",  # Altera -> Intel
        "AMDA": "SLDB",  # Amedica -> Solid Biosciences
        "AMRI": "AMRX",  # Albany Molecular Research
        "ANR": "ARLP",  # Alpha Natural Resources successor
        "ANTM": "ELV",  # Anthem -> Elevance Health
        "APHA": "TLRY",  # Aphria -> Tilray
        "BBRY": "BB",  # BlackBerry
        "BHI": "HAL",  # Baker Hughes -> Halliburton
        "BHGE": "BKR",  # Baker Hughes GE -> Baker Hughes
        "BRCD": "AVGO",  # Brocade -> Broadcom
        "BRCM": "AVGO",  # Broadcom Corp -> Broadcom
        "CERN": "ORCL",  # Cerner -> Oracle
        "CELG": "BMY",  # Celgene -> Bristol Myers Squibb
        "COH": "KORS",  # Coach -> Michael Kors
        "COV": "MDT",  # Covidien -> Medtronic
        "CTRX": "CNC",  # Centene acquisition
        "DISCA": "WBD",  # Discovery -> Warner Bros. Discovery
        "DISCK": "WBD",  # Discovery Class K -> Warner Bros. Discovery
        "DTV": "T",  # DirecTV -> AT&T
        "DPS": "KDP",  # Dr Pepper Snapple -> Keurig Dr Pepper
        "DWDP": "DD",  # DowDuPont -> Dow
        "EMC": "DELL",  # EMC -> Dell Technologies
        "ETFC": "MS",  # E*TRADE -> Morgan Stanley
        "FDC": "FTDR",  # First Data -> Fiserv
        "FDO": "DG",  # Family Dollar -> Dollar General
        "FLIR": "TDY",  # FLIR Systems -> Teledyne
        "FTR": "LUMN",  # Frontier -> Lumen successor
        "HCP": "DOC",  # HCP -> Healthpeak
        "HFC": "DINO",  # HollyFrontier -> HF Sinclair
        "HRS": "LHX",  # Harris -> L3Harris
        "JOY": "CAT",  # Joy Global -> Caterpillar
        "KORS": "CPRI",  # Michael Kors -> Capri
        "KRFT": "KHC",  # Kraft Foods -> Kraft Heinz
        "KSU": "CP",  # Kansas City Southern -> Canadian Pacific Kansas City
        "LBTYA": "LUMN",  # Historical Liberty-related security
        "LLL": "LHX",  # L3 Technologies -> L3Harris
        "LM": "BEN",  # Legg Mason -> Franklin Resources
        "LUK": "JEF",  # Leucadia -> Jefferies
        "LVLT": "LUMN",  # Level 3 -> CenturyLink/Lumen
        "MON": "BAYRY",  # Monsanto -> Bayer
        "MYL": "VTRS",  # Mylan -> Viatris
        "NBL": "CVX",  # Noble Energy -> Chevron
        "NBLX": "CVX",  # Noble Midstream -> Chevron
        "NUAN": "MSFT",  # Nuance -> Microsoft
        "NLOK": "GEN",  # NortonLifeLock -> Gen Digital
        "PBCT": "MTB",  # People's United -> M&T Bank
        "PXD": "XOM",  # Pioneer Natural Resources -> Exxon Mobil
        "RAI": "BTI",  # Reynolds American -> British American Tobacco
        "RHT": "IBM",  # Red Hat -> IBM
        "RTN": "RTX",  # Raytheon -> RTX
        "SIAL": "MRK",  # Sigma-Aldrich -> Merck
        "SPLK": "CSCO",  # Splunk -> Cisco
        "SQ": "XYZ",  # Block historical ticker
        "STJ": "ABT",  # St. Jude Medical -> Abbott
        "TWC": "CHTR",  # Time Warner Cable -> Charter
        "TWX": "WBD",  # Time Warner -> Warner Bros. Discovery
        "UTX": "RTX",  # United Technologies -> RTX
        "VIAB": "PARA",  # Viacom -> Paramount
        "VIAC": "PARA",  # ViacomCBS -> Paramount
        "WFM": "AMZN",  # Whole Foods -> Amazon
        "WLTW": "WTW",  # Willis Towers Watson
        "WPX": "APA",  # WPX Energy -> APA
        "WYN": "WH",  # Wyndham Worldwide -> Wyndham Hotels
        "ZNGA": "TTWO",  # Zynga -> Take-Two

        # ============================================================
        # CLEAN HISTORICAL TICKER CHANGES
        # ============================================================

        "ERTS": "EA",  # Electronic Arts: ERTS -> EA
        "BLL": "BALL",  # Ball: BLL -> BALL
        "CNHI": "CNH",  # CNH Industrial: CNHI -> CNH
        "TWTR": "X",  # Twitter -> X
        "BHGE": "BKR",  # Baker Hughes ticker transition

        # ============================================================
        # SUCCESSOR SECURITIES WE ARE INTENTIONALLY KEEPING
        # ============================================================

        "WRK": "SW",  # WestRock -> Smurfit WestRock
        "PSXP": "PSX",  # Phillips 66 Partners -> Phillips 66
        "RDS-A": "SHEL",  # Royal Dutch Shell A -> Shell
        "RDS-B": "SHEL",  # Royal Dutch Shell B -> Shell

        # ============================================================
        # OTHER HISTORICAL TICKER MAPPINGS ALREADY VERIFIED
        # ============================================================

        "AABA": "GOOG",
        "ATVI": "TTWO",  # NOTE: questionable; see below
    }

    STOOQ_FORMAT_ALIASES = {
        "BRK.B": "BRK-B",
        "BF.B": "BF-B",
        "CWEN.A": "CWEN-A",
        "BRK.A": "BRK-A",
        "HEI.A": "HEI-A",
        "LEN.B": "LEN-B",
        "BF.A": "BF-A",
        "HUB.B": "HUBB",
        "MOG.A": "MOG-A",
        "UA.C": "UAC",
    }

    stooq_file_map = {}

    # Build a lookup between ticker symbols and their local Stooq files
    for file_path in STOOQ_PATH.rglob("*.us.txt"):
        ticker = file_path.name[:-7].upper()
        stooq_file_map[ticker] = file_path

    # Historical aliases
    for old_ticker, new_ticker in STOOQ_TICKER_ALIASES.items():
        if new_ticker in stooq_file_map:
            stooq_file_map[old_ticker] = stooq_file_map[new_ticker]

    # Formatting aliases
    for old_ticker, new_ticker in STOOQ_FORMAT_ALIASES.items():
        if new_ticker in stooq_file_map:
            stooq_file_map[old_ticker] = stooq_file_map[new_ticker]

    return stooq_file_map


# Retrieves a singular security given a disclosure date and a timeframe to look back
def get_market_history(row, lookback_days=150):
    # Retrieve file path to selected ticker
    file_path = stooq_file_map.get(row["ticker"])

    # Return a missing value when historical data is unavailable
    if file_path is None:
        logging.warning(
            "No market data file found for ticker %s. Continuing prediction with nan values.",
            row["ticker"]
        )
        return None

    # Load in raw data and convert column names.
    stock_data = pd.read_csv(file_path)
    stock_data = stock_data.rename(columns={
        "<DATE>": "date",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<VOL>": "volume"
    })

    # Convert dates and count missing values
    stock_data["date"] = pd.to_datetime(stock_data["date"], errors="coerce")
    invalid_dates = stock_data["date"].isna().sum()
    invalid_values = stock_data[["open", "high", "low", "close", "volume"]].isna().sum().sum()

    # Log data quality issues
    validation_log(
        context=f"Raw market history data for: {row['ticker']} contains %s invalid dates and %s invalid values",
        invalid_dates=invalid_dates,
        invalid_values=invalid_values)

    # Sort chronologically and remove records without price data
    stock_data = stock_data.dropna(subset=["date"]).copy()
    history = stock_data[
        stock_data["date"] < row["disclosureDate"]
        ].sort_values("date")

    return history.tail(lookback_days)


# Log data quality issues identified during validation
def validation_log(context, invalid_dates=0, invalid_values=0, unknown_sizes=0):
    # Log invalid or missing dates when any are found.
    if invalid_dates > 0:
        logging.warning(
            "%s contains %s invalid or missing dates.",
            context,
            invalid_dates
        )

    # Log missing or invalid values when any are found.
    if invalid_values > 0:
        logging.warning(
            "%s contains %s invalid values.",
            context,
            invalid_values
        )

    # Log transactions with unknown reported sizes
    if unknown_sizes > 0:
        logging.warning(
            "%s contains %s transactions with unknown sizes.",
            context,
            unknown_sizes
        )


# =================================== Global configuration & Variables =================================================

# Configure application logging for validation and data quality messages
logging.basicConfig(
    filename="log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define paths to the datasets used throughout the data pipeline.
CONGRESSIONAL_DATA_PATH = Path("CSV_data") / "master_dataset.csv"
VIX_PATH = Path("CSV_data") / "VIX_data.csv"
STOOQ_PATH = Path("CSV_data") / "Stooq Market Data" / "daily" / "us"

# Build the ticker-to-file lookup used to retrieve historical stock data.
stooq_file_map = connect_stooq_db()
