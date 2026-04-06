import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import DAYS

def train_lasso_model(X, y, test_size=0.2, n_splits=5):
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=tscv, random_state=42, max_iter=20000)),
    ])

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test


def evaluate_model(y_train, y_test, y_pred_train, y_pred_test):
    train_metrics = {
        "R2": r2_score(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "MAE": mean_absolute_error(y_train, y_pred_train),
    }

    test_metrics = {
        "R2": r2_score(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE": mean_absolute_error(y_test, y_pred_test),
    }

    return train_metrics, test_metrics


def get_selected_features(model, feature_names):
    lasso = model.named_steps["lasso"]
    coef_series = pd.Series(lasso.coef_, index=feature_names)

    selected_features = (
        coef_series[coef_series != 0]
        .sort_values(key=lambda s: np.abs(s), ascending=False)
    )

    return selected_features

def bayesian_shrinkage_local(df, value_col, samples_col, window=7, k=20):
    df_copy = df.copy()

    # local mean (shift to avoid leakage)
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

def get_last_n_days_predictions(df, X, y, model, days=90):
    df_copy = df.copy()

    df_copy["date"] = pd.to_datetime(df_copy["date"])
    cutoff_date = df_copy["date"].max() - pd.Timedelta(days=days)

    mask = df_copy["date"] >= cutoff_date

    df_last = df_copy.loc[mask].reset_index(drop=True)
    X_last = X.loc[mask].reset_index(drop=True)
    y_last = y.loc[mask].reset_index(drop=True)

    y_pred_last = model.predict(X_last)

    results_df = pd.DataFrame({
        "date": df_last["date"],
        "actual": y_last,
        "predicted": y_pred_last
    })

    return results_df