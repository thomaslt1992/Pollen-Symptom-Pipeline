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

## Preprocessing

The pipeline includes automated preprocessing steps before model training:

- interpolation of pollen series (`POAC`, `birch`)
- Bayesian shrinkage of the symptom target
- lag creation for symptom and pollen variables
- rolling averages over past days

### Bayesian shrinkage for unreliable symptom observations

The target variable `averageOverallScoreWithMedication` may contain noisy daily values when the number of contributing patients is low.  
Instead of removing outliers, the pipeline applies a Bayesian-style shrinkage step using:

- the local historical mean of the symptom score
- the daily `samples` count as a reliability signal

Low-sample observations are shrunk more strongly toward the recent local mean, while high-sample observations remain closer to their original value.

This approach:
- keeps all observations
- reduces the influence of unstable extreme values
- avoids propagating noisy target values into lagged features

The shrinkage step is applied before lag and rolling-feature creation.

## Bayesian Shrinkage for Target Smoothing

The symptom target (`averageOverallScoreWithMedication`) can be noisy when the number of contributing patients (`samples`) is low.  
Instead of removing outliers, the pipeline applies a **Bayesian-style shrinkage** that adjusts each observation based on its reliability.

### Motivation

Daily symptom scores are estimates of an underlying true population value.  
When `samples` is small, the variance of this estimate is high:

![equation](https://latex.codecogs.com/png.latex?\mathrm{Var}(y_t)\propto\frac{1}{n_t})

where:
- \( y_t \) = observed symptom score at time \( t \)  
- \( n_t \) = number of samples (patients)

Low \( n_t \) → unreliable estimate → potential extreme values.

---

### Method

Each observation is shrunk toward a **local historical mean**:

\[
\mu_t = \frac{1}{w} \sum_{i=1}^{w} y_{t-i}
\]

where:
- \( \mu_t \) = local mean over a rolling window of size \( w \)  
- only past values are used to avoid leakage  

The adjusted value is computed as:

\[
\tilde{y}_t = \frac{n_t}{n_t + k} \cdot y_t + \frac{k}{n_t + k} \cdot \mu_t
\]

where:
- \( \tilde{y}_t \) = smoothed symptom score  
- \( n_t \) = number of samples  
- \( k \) = shrinkage parameter (controls smoothing strength)

---

### Interpretation

- If \( n_t \gg k \):  
  \[
  \tilde{y}_t \approx y_t
  \]
  → high confidence, minimal adjustment  

- If \( n_t \ll k \):  
  \[
  \tilde{y}_t \approx \mu_t
  \]
  → low confidence, strong smoothing  

---

### Properties

- preserves all observations (no outlier removal)  
- reduces noise driven by small sample sizes  
- adapts smoothing strength dynamically per time point  
- prevents propagation of noisy values into lag features  

---

### Pipeline Placement

This step is applied **before lag and rolling feature creation**, ensuring that all downstream features are based on a stabilized signal.

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

results = run_pipeline(
    data_dir="data",
    test_size=0.2,
    n_splits=5,
    lags=[1,2,3,5,7,10],
    windows=[3,5,7],
)

results["train_metrics"]
results["test_metrics"]
results["selected_features"]
```


## 6. Current Forbidden Columns

Make sure you adjust your forbidden columns (not included in the prediction input) according to your needs

```python
DEFAULT_FORBIDDEN_CURRENT = [
    "averageOverallScoreWithMedication",
    "standardDeviationWithMedication",
    "averageOverallScoreWithoutMedication",
    "standardDeviationWithoutMedication",
    "samples",
]
```