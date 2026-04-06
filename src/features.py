import numpy as np
import pandas as pd

from src.preprocessing import (
    bayesian_shrinkage_local,
    interpolate_column,
    create_lags,
    create_past_averages,
)


def prepare_features(df, lags, windows, forbidden_current):
    df = df.copy()

    df = interpolate_column(df, "poac")
    df = interpolate_column(df, "birch")

    df = bayesian_shrinkage_local(
        df,
        value_col="averageoverallscorewithmedication",
        samples_col="samples",
        window=7,
        k=20,
    )

    df = create_lags(df, "averageoverallscorewithmedication", lags)
    df = create_lags(df, "birch", lags)
    df = create_lags(df, "poac", lags)

    df = create_past_averages(df, "averageoverallscorewithmedication", windows)
    df = create_past_averages(df, "birch", windows)
    df = create_past_averages(df, "poac", windows)

    max_lag = max(lags + windows)
    df = df.iloc[max_lag:].reset_index(drop=True)

    df = df.loc[:, ~df.columns.duplicated()].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    target_col = "averageoverallscorewithmedication"
    y = df[target_col].copy()

    X = df.drop(columns=forbidden_current, errors="ignore").copy()
    X = X.drop(columns=["date"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    valid_idx = X.dropna().index.intersection(y.dropna().index)

    df_model = df.loc[valid_idx].reset_index(drop=True)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    return X, y, df_model