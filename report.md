# Housing Price Prediction — Analysis Report

This is a template report. Actual results vary by dataset configuration.

---

## 1. Dataset Overview

Load and inspect the dataset from the configured source (OpenML or CSV). Key statistics:

- Sample count, feature count, target variable name
- Descriptive statistics (`df.describe()`)
- Missing value check

---

## 2. Exploratory Data Analysis

### 2.1 Correlation Heatmap

![Correlation Heatmap](../output/boston/images/correlation_heatmap.png)

Shows which features are most correlated with the target variable, and reveals multicollinearity among predictors.

### 2.2 Target Distribution

![Target Distribution](../output/boston/images/target_distribution.png)

Histogram and boxplot showing the target variable's shape, central tendency, and outliers.

### 2.3 Top Features vs Target

![Feature Scatter](../output/boston/images/feature_scatter.png)

Scatter plots of the most correlated features against the target, with Pearson r annotations.

---

## 3. Linear Regression (Baseline)

Standard linear regression on standardized features as the baseline.

| Metric | Value |
|--------|-------|
| RMSE | varies |
| MAE | varies |
| R^2 | varies |

### 3.1 Standardized Coefficients

![Linear Coefficients](../output/boston/images/linear_coefficients.png)

### 3.2 Residual Analysis

![Residuals](../output/boston/images/residuals.png)

### 3.3 Predicted vs Actual

![Predicted vs Actual](../output/boston/images/predicted_vs_actual.png)

### 3.4 Fit Lines

![Linear Regression Fit](../output/boston/images/linear_regression_fit.png)

---

## 4. Multi-Model Comparison

| Model | Test RMSE | Test R^2 | CV RMSE (mean) |
|-------|-----------|----------|----------------|
| Gradient Boosting | | | |
| Random Forest | | | |
| Linear Regression | | | |
| Ridge | | | |
| ElasticNet | | | |
| Lasso | | | |

### 4.1 Feature Importance Comparison

![Multi-Algorithm Feature Importance](../output/boston/images/multi_algo_importance.png)

---

## 5. Best Model — Tuned Random Forest

| Parameter | Best Value |
|-----------|------------|
| `n_estimators` | varies |
| `max_depth` | varies |
| `min_samples_split` | varies |
| `min_samples_leaf` | varies |

| Metric | Value |
|--------|-------|
| CV RMSE | varies |
| Test RMSE | varies |
| Test R^2 | varies |
| Improvement over LR | varies |

### 5.1 Predicted vs Actual (Best Model)

![Best Model Predicted vs Actual](../output/boston/images/best_model_predicted_vs_actual.png)

### 5.2 Feature Importances

![RF Feature Importance](../output/boston/images/rf_feature_importance.png)

---

## 6. Conclusion

- Tree-based ensemble models consistently outperform linear methods on housing price data.
- Regularization provides minimal benefit when features are not highly collinear.
- The top 2-3 features dominate predictions; feature engineering may improve performance.
- Results and specific findings are printed to the console after each pipeline run.
