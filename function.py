import os
import random
from collections import deque
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Set a fixed random seed to ensure reproducible and consistent results across runs
np.random.seed(314)
random.seed(314)


def shuffle_in_unison(a, b):
    """
    Shuffles two NumPy arrays simultaneously so that the row-by-row mapping
    between features (a) and labels (b) is perfectly preserved.
    """
    # Capture the internal state of the random number generator
    state = np.random.get_state()
    # Randomly shuffle the feature array in-place
    np.random.shuffle(a)
    # Restore the exact same random state to synchronize the next shuffle
    np.random.set_state(state)
    # Shuffle the target array using the exact same permutation order
    np.random.shuffle(b)


def load_data(
    ticker,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    split_by_date=True,
    test_size=0.2,
    feature_columns=["adjclose", "volume", "open", "high", "low"],
    start_date="2020-01-01",  # Requirement 1a: Custom start date parameter
    end_date="2023-08-01",  # Requirement 1a: Custom end date parameter
    load_local=True,  # Requirement 1d: Toggle to load data from a local file
    store_local=True,  # Requirement 1d: Toggle to save downloaded data locally
):
    """
    Loads stock data, handles missing values, normalizes features, caches files,
    and splits the dataset into training and testing sets based on user preferences.
    """
    # Define a clean, dynamic filename based on the asset ticker and date range
    local_filename = f"data/{ticker}_{start_date}_to_{end_date}.csv"

    # Requirement 1d: Check if local loading is enabled and the file exists on disk
    if load_local and os.path.exists(local_filename):
        print(f"[INFO] Loading dataset locally from: {local_filename}")
        # Read the local CSV file and parse its index column as date objects
        df = pd.read_csv(local_filename, index_col=0, parse_dates=True)
    else:
        print(f"[INFO] Fetching data live from Yahoo Finance for: {ticker}")
        # Requirement 1a & 1d: Download live data via the API using explicit date boundaries
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)

        # Standardize all column headers to lowercase strings to eliminate case mismatch bugs
        df.columns = df.columns.str.lower()

        # Clone the standard "close" price array into an "adjclose" column to support legacy logic
        df["adjclose"] = df["close"]

        # Requirement 1d: If local saving is enabled, ensure directory exists and write out the file
        if store_local:
            if not os.path.isdir("data"):
                os.mkdir("data")
            # Export the pristine, unscaled DataFrame out to a static CSV file
            df.to_csv(local_filename)

    # Initialize a dictionary to act as the main data carrier container
    result = {}
    # Keep an untouched, unscaled copy of the full DataFrame for final backtesting calculations
    result["df"] = df.copy()

    # Verify that every single requested feature column actually exists in the working DataFrame
    for col in feature_columns:
        assert (
            col in df.columns
        ), f"'{col}' does not exist in the dataframe."  # Halt if a feature is missing

    # Convert the index dates into an explicit data column if it doesn't already exist
    if "date" not in df.columns:
        df["date"] = df.index

    # Requirement 1e: Scaler data structure and feature scaling execution
    if scale:
        # Create an internal dictionary to map feature names to their respective scaling models
        column_scaler = {}
        for column in feature_columns:
            # Instantiate an independent MinMaxScaler to force values into a strict (0, 1) boundary
            scaler = preprocessing.MinMaxScaler() 
            # Learn the Min and Max values ​​of that data column, perform a conversion calculation to the range (0, 1), 
            # and then assign the cleaned data back to overwrite the old column in the df table.
            df[column] = scaler.fit_transform( 
                np.expand_dims(df[column].values, axis=1) 
                 # Take the 1D array of values ​​from the current column
                 # and use the expand_dims function to insert a new axis at the column position (axis=1), 
                 # transforming the array from a one-dimensional array into a vertical 2D column array, 
                 # because the fit_transform function requires the input data to be a 2D array.
            )
            # Store the trained scaler instance linked to its column string name for future reference
            column_scaler[column] = scaler
        # Expose the dictionary of scalers out to the main result object
        result["column_scaler"] = column_scaler

    # Requirement 1b: Create the target column by shifting the price backwards
    df["future"] = df["adjclose"].shift(-lookup_step)

    # Temporarily isolate the final few feature records before they are removed by dropna
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # Requirement 1b: Drop any incomplete rows containing NaN cells generated by shifting
    df.dropna(inplace=True)

    # Initialize lists to process and build historical sequence blocks
    sequence_data = []
    # Create a double-ended queue with a locked maximum length to act as a moving window
    sequences = deque(maxlen=n_steps)

    # Iterate simultaneously over rows of combined feature lists and target labels
    for entry, target in zip(
        df[feature_columns + ["date"]].values, df["future"].values
    ):
        # Push the current row item into the right side of the sliding queue window
        sequences.append(entry)
        # Once the queue gathers enough consecutive history days, save the sequence block
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Construct the ultimate prediction seed array by concatenating the window tail with the lookahead gap
    last_sequence = list([s[: len(feature_columns)] for s in sequences]) + list(
        last_sequence
    )
    last_sequence = np.array(last_sequence).astype(np.float32)
    result["last_sequence"] = last_sequence

    # Separate out the prepared sequence datasets into explicit feature inputs (X) and labels (y)
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # Transform standard Python lists into raw, highly optimized NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Requirement 1c: Handle different data splitting strategies (Chronological vs Random)
    if split_by_date:
        # Calculate the hard boundary index dividing training records from testing records
        train_samples = int((1 - test_size) * len(X))
        # Assign historical older blocks to the training matrices
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        # Assign more recent blocks to the independent testing matrices
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        # If shuffling is enabled, randomize the internal sequence layout order within each set
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # Perform an unconditional, non-linear random data split using scikit-learn
        (
            result["X_train"],
            result["X_test"],
            result["y_train"],
            result["y_test"],
        ) = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # Extract the original index timestamp references representing the testing blocks
    dates = result["X_test"][:, -1, -1]
    # Reconstruct a clean validation DataFrame tracking only test dates
    result["test_df"] = result["df"].loc[dates]
    # Filter out any redundant date indexes to keep validation measurements mathematically sound
    result["test_df"] = result["test_df"][
        ~result["test_df"].index.duplicated(keep="first")
    ]

    # Strip out trailing non-numeric date tags from the arrays and enforce standard float32 precision
    result["X_train"] = result["X_train"][:, :, : len(feature_columns)].astype(
        np.float32
    )
    result["X_test"] = result["X_test"][:, :, : len(feature_columns)].astype(
        np.float32
    )

    # Return the fully processed dictionary containing all datasets, dataframes, and scalers
    return result