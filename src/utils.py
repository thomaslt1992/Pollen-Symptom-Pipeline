import matplotlib.pyplot as plt
import shap
import random
import pandas as pd
from src.config import POLLEN_SEASON_COLORS, DAYS

def save_metrics(filepath, train_metrics, test_metrics):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("TRAIN METRICS\n")
        for k, v in train_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nTEST METRICS\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")


def save_selected_features(filepath, selected_features):
    df = selected_features.rename("coefficient").reset_index()
    df.columns = ["feature", "coefficient"]
    df.to_csv(filepath, index=False)

def plot_forecast(results_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results_df["date"], results_df["actual"], label="Actual")
    ax.plot(results_df["date"], results_df["predicted"], label="Predicted")

    season_cols = [col for col in results_df.columns if col.endswith("_in_season")]
    used_labels = set()

    for season_col in season_cols:
        pollen_name = season_col.replace("_in_season", "")
        color = POLLEN_SEASON_COLORS.get(
            pollen_name,
            (random.random(), random.random(), random.random())
        )

        in_season = results_df[season_col].fillna(0).astype(int).values
        dates = results_df["date"].values

        start_idx = None
        for i, val in enumerate(in_season):
            if val == 1 and start_idx is None:
                start_idx = i
            elif val == 0 and start_idx is not None:
                label = f"{pollen_name} season" if pollen_name not in used_labels else None
                ax.axvspan(dates[start_idx], dates[i - 1], alpha=0.15, color=color, label=label)
                used_labels.add(pollen_name)
                start_idx = None

        if start_idx is not None:
            label = f"{pollen_name} season" if pollen_name not in used_labels else None
            ax.axvspan(dates[start_idx], dates[len(dates) - 1], alpha=0.15, color=color, label=label)
            used_labels.add(pollen_name)

    ax.set_title(f"Last {DAYS} as Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Symptom Score")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_shap_lasso_summary(model, X_train, X_test, max_display=15):
    scaler = model.named_steps["scaler"]
    lasso = model.named_steps["lasso"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    explainer = shap.Explainer(lasso, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_shap_waterfall(model, X_train, X_sample, idx=0):

    scaler = model.named_steps["scaler"]
    lasso = model.named_steps["lasso"]

    X_train_scaled = scaler.transform(X_train)
    X_sample_scaled = scaler.transform(X_sample)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_sample_scaled = pd.DataFrame(X_sample_scaled, columns=X_sample.columns)

    explainer = shap.Explainer(lasso, X_train_scaled)
    shap_values = explainer(X_sample_scaled)

    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_for_all_in_season_periods(model, X_train, X_test, max_display=15):
    scaler = model.named_steps["scaler"]
    lasso = model.named_steps["lasso"]

    season_cols = [col for col in X_test.columns if col.endswith("_in_season")]

    if not season_cols:
        print("No *_in_season columns found in X_test.")
        return {}

    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    explainer = shap.Explainer(lasso, X_train_scaled)

    shap_results = {}

    for season_col in season_cols:
        X_subset = X_test[X_test[season_col] == 1].copy()

        if X_subset.empty:
            print(f"Skipping {season_col}: no rows with value 1.")
            continue

        X_subset_scaled = scaler.transform(X_subset)
        X_subset_scaled = pd.DataFrame(
            X_subset_scaled,
            columns=X_subset.columns,
            index=X_subset.index,
        )

        shap_values = explainer(X_subset_scaled)
        shap_results[season_col] = shap_values

        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.title(f"{season_col} - SHAP Beeswarm")
        plt.tight_layout()
        plt.show()

        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title(f"{season_col} - SHAP Bar")
        plt.tight_layout()
        plt.show()

    return shap_results