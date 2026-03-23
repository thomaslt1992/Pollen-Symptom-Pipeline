import argparse
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import save_metrics, save_selected_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pollen symptom forecasting pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing input CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for TimeSeriesSplit",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 7],
        help="Lag values to create, e.g. --lags 1 2 3 5 7",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Rolling window sizes, e.g. --windows 3 5 7",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_pipeline(
        data_dir=args.data_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        lags=args.lags,
        windows=args.windows,
    )

    train_metrics = results["train_metrics"]
    test_metrics = results["test_metrics"]
    selected_features = results["selected_features"]

    print("TRAIN METRICS")
    for metric_name, metric_value in train_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nTEST METRICS")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nSELECTED FEATURES")
    print(selected_features)

    save_metrics(output_dir / "metrics.txt", train_metrics, test_metrics)
    save_selected_features(output_dir / "selected_features.csv", selected_features)


if __name__ == "__main__":
    main()