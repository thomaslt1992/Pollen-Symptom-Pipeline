# Pollen–Symptom Forecasting Pipeline

This repository contains a lightweight machine learning pipeline for modeling the relationship between airborne pollen concentrations and reported symptom severity.

The pipeline:
- preprocesses pollen and symptom data
- creates time-based features (lags, rolling averages)
- trains a baseline Lasso regression model
- performs automatic feature selection
- evaluates model performance on a time-based split

This is an evolving project. Future updates will include:
- additional models (tree-based, neural networks)
- improved feature engineering
- visualizations and exploratory analysis

---

## Project Structure

src/
    pipeline.py        # main pipeline logic
    features.py        # feature engineering
    preprocessing.py   # interpolation, lags, rolling features
    model.py           # training + evaluation
    data_loader.py     # data loading / merging
    utils.py           # saving outputs
    config.py          # default parameters

main.py                # CLI entry point
pipeline.ipynb         # optional notebook usage
data/                  # (ignored, local only)
outputs/               # saved results

---

## How to Run (Command Line)

From the project root:

python main.py --data-dir data --output-dir outputs

Optional arguments:

python main.py \
    --data-dir data \
    --output-dir outputs \
    --test-size 0.2 \
    --n-splits 5

Outputs:
- metrics (R², RMSE, MAE)
- selected features (from Lasso)

---

## How to Use in a Notebook

from src.pipeline import run_pipeline

results = run_pipeline(data_dir="data")

results["train_metrics"]
results["test_metrics"]
results["selected_features"]

---

## Notes

- The data/ folder is excluded from version control (confidential data)
- The pipeline assumes time-indexed data with a "date" column
- Missing values are handled via interpolation