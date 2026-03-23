# Pollen–Symptom Forecasting Pipeline

This repository contains a lightweight machine learning pipeline for modeling the relationship between airborne pollen concentrations and reported symptom severity.

## Overview

The pipeline:
- preprocesses pollen and symptom data  
- creates time-based features (lags, rolling averages)  
- trains a baseline Lasso regression model  
- performs automatic feature selection  
- evaluates model performance using a time-based split  

This is an evolving project. Planned extensions include:
- additional models (tree-based, neural networks)  
- improved feature engineering  
- visualizations and exploratory analysis  

---

## Project Structure
```text
src/
│
├── pipeline.py # main pipeline logic
├── features.py # feature engineering
├── preprocessing.py # interpolation, lags, rolling features
├── model.py # training and evaluation
├── data_loader.py # data loading / merging
├── utils.py # saving outputs
└── config.py # default parameters

main.py # CLI entry point
pipeline.ipynb # optional notebook usage

data/ # local (ignored, confidential)
outputs/ # saved results
```
---

## How to Run (Command Line)

From the project root:

```bash
python main.py --data-dir data --output-dir outputs
```

## Optional arguments
```
python main.py \
  --data-dir data \
  --output-dir outputs \
  --test-size 0.2 \
  --n-splits 5
```

Outputs
metrics (R², RMSE, MAE)
selected features (from Lasso)

## How to use in a notebook

```
from src.pipeline import run_pipeline

results = run_pipeline(data_dir="data")

results["train_metrics"]
results["test_metrics"]
results["selected_features"]
```
