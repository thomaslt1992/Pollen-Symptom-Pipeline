from sklearn.neighbors import NearestNeighbors
import numpy as np

def bayesian_shrinkage_local(df, value_col, samples_col, window=7, k=20):
    df_copy = df.copy()

    local_mean = (
        df_copy[value_col]
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
    )

    n = df_copy[samples_col].fillna(0)
    weight = n / (n + k)

    df_copy[value_col] = weight * df_copy[value_col] + (1 - weight) * local_mean

    return df_copy

def knn_fill_na(df, column, n_neighbors=5):

    df_copy = df.copy()
    values = df_copy[column].values.reshape(-1, 1)

    valid_mask = ~np.isnan(values[:,0])

    X_train = values[valid_mask]
    X_missing = values[~valid_mask]

    if len(X_missing) == 0:
        return df_copy

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_train)

    distances, indices = knn.kneighbors(X_missing)
    imputed_values = X_train[indices].mean(axis=1)
    df_copy.loc[~valid_mask, column] = imputed_values

    return df_copy


def create_lags(df, column, lags):
    """
    Create lagged versions of a column.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column to lag
    lags : list[int]
        List of lag values
    """

    df_copy = df.copy()

    for lag in lags:
        df_copy[f"{column}_lag_{lag}"] = df_copy[column].shift(lag)

    return df_copy


def create_past_averages(df, column, windows):
    """
    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column to compute averages from
    windows : list[int]
    """

    df_copy = df.copy()

    for w in windows:
        df_copy[f"{column}_avg_prev_{w}"] = (
            df_copy[column]
            .shift(1)
            .rolling(window=w)
            .mean()
        )
    return df_copy

def interpolate_column(df, column, date_col="date"):
    df_copy = df.copy()

    if date_col in df_copy.columns:
        df_copy = df_copy.sort_values(date_col).reset_index(drop=True)

    df_copy[column] = df_copy[column].interpolate(method="linear", limit_direction="both")

    return df_copy
