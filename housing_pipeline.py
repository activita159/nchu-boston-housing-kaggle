import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11


def load_config():
    parser = argparse.ArgumentParser(description='Housing Price Prediction Pipeline')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config JSON file (e.g. config/boston.json)')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


def load_data(cfg):
    data_cfg = cfg['data']
    source = data_cfg.get('source', 'openml')

    if source == 'openml':
        name = data_cfg.get('openml_name', 'boston')
        version = data_cfg.get('openml_version', 1)
        print(f"Fetching dataset from OpenML: name='{name}', version={version}...")
        ds = fetch_openml(name=name, version=version, as_frame=True, parser='auto')
        df = ds.data
        target_col = data_cfg.get('target_column', 'MEDV')
        df[target_col] = ds.target
        feature_names = list(ds.feature_names)
    elif source == 'csv':
        csv_path = data_cfg['csv_path']
        target_col = data_cfg['target_column']
        print(f"Loading CSV: {csv_path}...")
        df = pd.read_csv(csv_path)
        feature_names = data_cfg.get('feature_columns', [c for c in df.columns if c != target_col])
        if not feature_names:
            feature_names = [c for c in df.columns if c != target_col]
    else:
        raise ValueError(f"Unsupported data source: {source}")

    return df, feature_names, target_col


def main():
    cfg = load_config()
    project_name = cfg['project_name']
    target_desc = cfg['data'].get('target_desc', cfg['data']['target_column'])
    target_unit = cfg['data'].get('target_unit', '')
    dataset_source = cfg['data'].get('dataset_source', '')
    dataset_desc = cfg['data'].get('dataset_desc', '')
    target_display = cfg['data'].get('target_display', cfg['data']['target_column'])
    output_dir = cfg['output']['dir']
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    train_cfg = cfg.get('training', {})
    test_size = train_cfg.get('test_size', 0.2)
    random_state = train_cfg.get('random_state', 42)
    cv_folds = train_cfg.get('cv_folds', 5)

    # ============================================================
    # 1. FETCH DATA
    # ============================================================
    print("=" * 60)
    print(f"STEP 1: Loading dataset — {project_name}...")
    print("=" * 60)

    df, feature_names, target_col = load_data(cfg)

    print(f"Dataset shape: {df.shape[0]} samples x {df.shape[1]} columns")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Target: {target_col} ({target_desc})")
    print(f"Source: {dataset_source}")
    print()

    # ============================================================
    # 2. EDA & VISUALIZATION
    # ============================================================
    print("STEP 2: Exploratory Data Analysis...")

    # 2a. Basic stats
    print("\n--- Basic Statistics ---")
    print(df.describe().round(2).to_string())
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")

    # 2b. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title(f'{project_name} — Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/correlation_heatmap.png")

    # 2c. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[target_col], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {df[target_col].mean():.2f}')
    axes[0].axvline(df[target_col].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median = {df[target_col].median():.2f}')
    axes[0].set_xlabel(f'{target_display} ({target_unit})' if target_unit else target_display)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {target_display}')
    axes[0].legend()

    sns.boxplot(x=df[target_col], ax=axes[1], color='lightcoral')
    axes[1].set_title(f'Boxplot of {target_display}')
    axes[1].set_xlabel(f'{target_display} ({target_unit})' if target_unit else target_display)
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'target_distribution.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/target_distribution.png")

    # 2d. Top features vs target scatter
    top_n = cfg.get('visualization', {}).get('top_features_count', 4)
    top_features = corr[target_col].abs().sort_values(ascending=False).index[1:top_n + 1]
    n_feat = len(top_features)
    n_cols = 2
    n_rows = (n_feat + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 5 * n_rows))
    axes = axes.flatten()
    for i, feat in enumerate(top_features):
        axes[i].scatter(df[feat], df[target_col], alpha=0.5, color='teal', edgecolors='none')
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(f'{target_display} ({target_unit})' if target_unit else target_display)
        axes[i].set_title(f'{feat} vs {target_display} (r = {corr.loc[feat, target_col]:.3f})')
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'feature_scatter.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/feature_scatter.png")

    # ============================================================
    # 3. PREPROCESSING
    # ============================================================
    print("\nSTEP 3: Preprocessing...")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================================
    # 4. LINEAR REGRESSION (BASELINE)
    # ============================================================
    print("\nSTEP 4: Linear Regression Baseline...")

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"   RMSE: {rmse_lr:.4f}")
    print(f"   MAE:  {mae_lr:.4f}")
    print(f"   R^2:   {r2_lr:.4f}")

    # 4a. Linear Regression Coefficients
    fig, ax = plt.subplots(figsize=(12, 6))
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)

    colors = ['#d9534f' if c < 0 else '#5cb85c' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardized Coefficient')
    ax.set_ylabel('Feature')
    ax.set_title(f'{project_name} — Standardized Coefficients', fontsize=13, fontweight='bold')
    for i, (feat, val) in enumerate(zip(coef_df['Feature'], coef_df['Coefficient'])):
        ax.text(val + (0.2 if val >= 0 else -0.2), i, f'{val:.2f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'linear_coefficients.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/linear_coefficients.png")

    # 4b. Residual Plot
    residuals = y_test.values - y_pred_lr
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred_lr, residuals, alpha=0.5, color='steelblue', edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel(f'Fitted Values (Predicted {target_display})')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted Values')

    axes[1].hist(residuals, bins=25, color='mediumseagreen', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')

    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'residuals.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/residuals.png")

    # 4c. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_lr, alpha=0.5, color='steelblue', edgecolors='none', s=50)
    lims = [
        min(y_test.min(), y_pred_lr.min()) - 2,
        max(y_test.max(), y_pred_lr.max()) + 2
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction (y = x)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Actual {target_display} ({target_unit})' if target_unit else f'Actual {target_display}')
    ax.set_ylabel(f'Predicted {target_display} ({target_unit})' if target_unit else f'Predicted {target_display}')
    ax.set_title(f'{project_name} — Predicted vs Actual\nRMSE = {rmse_lr:.3f} | R^2 = {r2_lr:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'predicted_vs_actual.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/predicted_vs_actual.png")

    # 4d. Linear Regression Fit Lines
    viz_cfg = cfg.get('visualization', {})
    fit_features = viz_cfg.get('fit_line_features')
    if fit_features is None:
        fit_features = list(corr[target_col].abs().sort_values(ascending=False).index[1:3])

    n_fit = len(fit_features)
    fig, axes = plt.subplots(1, n_fit, figsize=(7 * n_fit, 6))
    if n_fit == 1:
        axes = [axes]

    for ax, feat in zip(axes, fit_features):
        X_feat = df[[feat]].values
        y_feat = df[target_col].values
        lr_single = LinearRegression()
        lr_single.fit(X_feat, y_feat)
        y_line = lr_single.predict(X_feat)
        r2 = lr_single.score(X_feat, y_feat)
        slope = lr_single.coef_[0]
        intercept = lr_single.intercept_
        sign = '+' if intercept >= 0 else '-'

        ax.scatter(X_feat, y_feat, alpha=0.4, color='steelblue', edgecolors='none', s=40)
        ax.plot(X_feat, y_line, color='#d9534f', linewidth=2.5,
                label=f'y = {slope:.2f}x {sign} {abs(intercept):.2f}')
        ax.set_xlabel(feat)
        ax.set_ylabel(f'{target_display} ({target_unit})' if target_unit else target_display)
        ax.set_title(f'{feat} vs {target_display} — Linear Regression Fit\nR^2 = {r2:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'linear_regression_fit.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/linear_regression_fit.png")

    # ============================================================
    # 5. MODEL COMPARISON
    # ============================================================
    print("\nSTEP 5: Multi-Model Comparison...")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (alpha=1.0)': Ridge(alpha=1.0),
        'Lasso (alpha=0.1)': Lasso(alpha=0.1, max_iter=10000),
        'ElasticNet (alpha=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append({
            'Model': name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R^2': r2_score(y_test, y_pred),
        })

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds,
                                    scoring='neg_root_mean_squared_error')
        results[-1]['CV RMSE (mean)'] = -cv_scores.mean()
        results[-1]['CV RMSE (std)'] = cv_scores.std()

    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("\n" + results_df.to_string(index=False))

    # ============================================================
    # 5b. Multi-Algorithm Feature Importance Comparison
    # ============================================================
    print("\nSTEP 5b: Multi-Algorithm Feature Importance Comparison...")

    lr_importance = np.abs(lr.coef_)
    lr_importance_norm = lr_importance / lr_importance.sum()

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_importance = rf_model.feature_importances_
    rf_importance_norm = rf_importance / rf_importance.sum()

    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state)
    gb_model.fit(X_train_scaled, y_train)
    gb_importance = gb_model.feature_importances_
    gb_importance_norm = gb_importance / gb_importance.sum()

    avg_importance = (lr_importance_norm + rf_importance_norm + gb_importance_norm) / 3
    sort_idx = np.argsort(avg_importance)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

    algo_names = ['Linear Regression\n(|coef|)', 'Random Forest', 'Gradient Boosting']
    algo_importances = [lr_importance_norm, rf_importance_norm, gb_importance_norm]
    algo_colors = ['#5bc0de', '#5cb85c', '#f0ad4e']

    for ax, name, imp, color in zip(axes, algo_names, algo_importances, algo_colors):
        ax.barh(np.array(feature_names)[sort_idx], imp[sort_idx], color=color, edgecolor='white')
        ax.set_xlabel('Normalized Importance')
        ax.set_title(name, fontsize=12, fontweight='bold')
        for i, val in enumerate(imp[sort_idx]):
            ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=8)

    fig.suptitle(f'{project_name} — Multi-Algorithm Feature Importance', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'multi_algo_importance.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/multi_algo_importance.png")

    # ============================================================
    # 6. HYPERPARAMETER TUNING (Best Model)
    # ============================================================
    print("\nSTEP 6: Hyperparameter Tuning (Random Forest)...")

    default_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    param_grid = cfg.get('models', {}).get('rf_tune', default_param_grid)

    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_grid, cv=cv_folds, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train_scaled, y_train)
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   Best CV RMSE: {-rf_grid.best_score_:.4f}")

    best_model = rf_grid.best_estimator_
    y_pred_best = best_model.predict(X_test_scaled)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)
    print(f"   Test RMSE: {rmse_best:.4f}")
    print(f"   Test R^2:   {r2_best:.4f}")

    # ============================================================
    # 7. BEST MODEL: Predicted vs Actual
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_best, alpha=0.5, color='darkorange', edgecolors='none', s=50)
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Actual {target_display} ({target_unit})' if target_unit else f'Actual {target_display}')
    ax.set_ylabel(f'Predicted {target_display} ({target_unit})' if target_unit else f'Predicted {target_display}')
    ax.set_title(f'{project_name} — Best Model (Tuned RF): Predicted vs Actual\n'
                 f'RMSE = {rmse_best:.3f} | R^2 = {r2_best:.3f}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'best_model_predicted_vs_actual.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/best_model_predicted_vs_actual.png")

    # Feature importance from best model
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)
    ax.barh(np.array(feature_names)[indices], importances[indices], color='steelblue', edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'{project_name} — Random Forest Feature Importances', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(images_dir, 'rf_feature_importance.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {images_dir}/rf_feature_importance.png")

    # ============================================================
    # 8. SUBMISSION
    # ============================================================
    print("\nSTEP 8: Generating submission.csv...")

    id_col = cfg['output'].get('submission_id_col', 'ID')
    target_out_col = cfg['output'].get('submission_target_col', target_col)

    submission = pd.DataFrame({
        id_col: range(1, len(y_test) + 1),
        target_out_col: y_pred_best
    })
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"   Saved: {submission_path} ({len(submission)} rows)")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Project:         {project_name}")
    print(f"   Dataset:         {df.shape[0]} samples, {len(feature_names)} features")
    print(f"   Train / Test:    {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"   Linear Reg RMSE: {rmse_lr:.4f}  R^2: {r2_lr:.4f}")
    print(f"   Best Model RMSE: {rmse_best:.4f}  R^2: {r2_best:.4f}")
    print(f"   Improvement:     {(rmse_lr - rmse_best) / rmse_lr * 100:.2f}% RMSE reduction")
    print(f"   All images saved to {images_dir}/")
    print(f"   Submission saved to {submission_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
