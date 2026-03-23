import numpy as np
import pandas as pd

from src.preprocessing import interpolate_column, create_lags, create_past_averages
from src.preprocessing import bayesian_shrinkage_local

def prepare_features(df, lags, windows, forbidden_current):
    df = df.copy()

    df = interpolate_column(df, "POAC")
    df = interpolate_column(df, "birch")

    #This will overwrite the original columns, but that's fine since we won't be using them directly in the model. 
    # The smoothed versions will be more informative and less noisy for feature engineering.
    df = bayesian_shrinkage_local(df,value_col="averageOverallScoreWithMedication",samples_col="samples",window=7,k=20,)

    df = create_lags(df, "averageOverallScoreWithMedication", lags)
    df = create_lags(df, "birch", lags)
    df = create_lags(df, "POAC", lags)

    df = create_past_averages(df, "averageOverallScoreWithMedication", windows)
    df = create_past_averages(df, "birch", windows)
    df = create_past_averages(df, "POAC", windows)

    max_lag = max(lags + windows)
    df = df.iloc[max_lag:].reset_index(drop=True)

    df = df.loc[:, ~df.columns.duplicated()].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    target_col = "averageOverallScoreWithMedication"
    y = df[target_col].copy()

    X = df.drop(columns=forbidden_current, errors="ignore").copy()
    X = X.drop(columns=["date"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    return X, y