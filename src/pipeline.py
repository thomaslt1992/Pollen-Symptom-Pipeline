from pathlib import Path

from src.config import (
    DEFAULT_FORBIDDEN_CURRENT,
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    DEFAULT_SELECTED_POLLEN,
    DAYS,
)
from src.data_loader import load_and_merge_data
from src.features import prepare_features
from src.model import evaluate_model, get_selected_features, train_lasso_model
from src.model import get_last_n_days_predictions

def run_pipeline(
    data_dir="data",
    test_size=0.2,
    n_splits=5,
    lags=DEFAULT_LAGS,
    windows=DEFAULT_WINDOWS,
    forbidden_current=DEFAULT_FORBIDDEN_CURRENT,
    selected_pollen=DEFAULT_SELECTED_POLLEN,
):
    data_dir = Path(data_dir)

    df = load_and_merge_data(
        data_dir=data_dir,
        selected_pollen=selected_pollen,
    )

    X, y, df_model = prepare_features(
    df=df,
    lags=lags,
    windows=windows,
    forbidden_current=forbidden_current)

    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_lasso_model(
        X=X,
        y=y,
        test_size=test_size,
        n_splits=n_splits,
    )

    train_metrics, test_metrics = evaluate_model(
        y_train=y_train,
        y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
    )

    selected_features = get_selected_features(model, X_train.columns)

    forecast_df = get_last_n_days_predictions(
    df=df_model,
    X=X,
    y=y,
    model=model,
    days=DAYS,)

    return {
    "df": df,
    "df_model": df_model,
    "X": X,
    "y": y,
    "model": model,
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "y_pred_train": y_pred_train,
    "y_pred_test": y_pred_test,
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "selected_features": selected_features,
    "forecast_df": forecast_df,}