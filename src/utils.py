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