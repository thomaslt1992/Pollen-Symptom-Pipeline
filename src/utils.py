import matplotlib.pyplot as plt

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
    plt.figure(figsize=(12, 6))

    plt.plot(results_df["date"], results_df["actual"], label="Actual")
    plt.plot(results_df["date"], results_df["predicted"], label="Predicted")

    plt.title("Last 3 Months Forecast")
    plt.xlabel("Date")
    plt.ylabel("Symptom Score")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()