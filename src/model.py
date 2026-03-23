import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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