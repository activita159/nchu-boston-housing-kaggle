# Boston Housing Price Prediction

Predicting median home values in Boston suburbs using sklearn regression models.

![Poster](images/poster.png)

## Dataset

- **Source**: [OpenML — Boston Housing](https://www.openml.org/d/531) (506 samples, 13 features)
- **Target**: `MEDV` — Median value of owner-occupied homes in $1000s

## Project Structure

```
.
├── boston_housing.py              # One-click pipeline
├── images/                        # All generated plots
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│   ├── feature_scatter.png
│   ├── linear_coefficients.png
│   ├── predicted_vs_actual.png
│   ├── residuals.png
│   ├── best_model_predicted_vs_actual.png
│   └── rf_feature_importance.png
├── report.md                      # Full analysis report
├── submission.csv                 # Kaggle-format predictions
└── README.md                      # This file
```

## Quick Start

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python boston_housing.py
```

## Results

| Model | RMSE | R^2 |
|-------|------|-----|
| Gradient Boosting | 2.445 | 0.918 |
| Random Forest | 2.915 | 0.884 |
| Linear Regression | 4.929 | 0.669 |
| Ridge | 4.931 | 0.668 |
| ElasticNet | 5.020 | 0.656 |
| Lasso | 5.065 | 0.650 |

**Best Model** — Tuned Random Forest (`max_depth=10, n_estimators=100, min_samples_leaf=2`)

- **Test RMSE**: 3.024
- **Test R^2**: 0.875
- **Improvement**: 38.64% over Linear Regression

## Key Features

| Feature | Description |
|---------|-------------|
| RM | Average number of rooms per dwelling |
| LSTAT | % lower status of the population |
| PTRATIO | Pupil-teacher ratio by town |
| DIS | Weighted distances to employment centers |
| NOX | Nitric oxides concentration |
