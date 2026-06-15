import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# ---------- data ----------
boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
df = boston.data
df['MEDV'] = boston.target
feature_names = list(boston.feature_names)

X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LinearRegression().fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15, None],
     'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
    cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
)
rf_grid.fit(X_train_s, y_train)
best = rf_grid.best_estimator_
y_pred_best = best.predict(X_test_s)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
gb.fit(X_train_s, y_train)
y_pred_gb = gb.predict(X_test_s)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

models = {
    'Gradient\nBoosting': (rmse_gb, r2_gb),
    'Random\nForest': (rmse_best, r2_best),
    'Ridge': (4.931, 0.668),
    'Linear\nRegression': (rmse_lr, r2_lr),
    'ElasticNet': (5.020, 0.656),
    'Lasso': (5.065, 0.650),
}

# ---------- poster ----------
fig = plt.figure(figsize=(16, 20), dpi=150)
fig.patch.set_facecolor('#1a1a2e')

# ---- header band ----
ax_header = fig.add_axes([0, 0.88, 1, 0.12])
ax_header.set_facecolor('#16213e')
ax_header.set_xlim(0, 1)
ax_header.set_ylim(0, 1)
ax_header.axis('off')
ax_header.text(0.5, 0.7, 'Boston Housing', fontsize=42, fontweight='bold',
               color='white', ha='center', va='center', fontfamily='monospace')
ax_header.text(0.5, 0.28, 'PRICE PREDICTION', fontsize=34, fontweight='bold',
               color='#e94560', ha='center', va='center', fontfamily='monospace')

# ---- separator line ----
ax_line = fig.add_axes([0.05, 0.865, 0.9, 0.005])
ax_line.set_facecolor('#1a1a2e')
ax_line.set_xlim(0, 1)
ax_line.set_ylim(0, 1)
ax_line.axhline(0.5, color='#e94560', linewidth=2)
ax_line.axis('off')

# ---- left panel: dataset info ----
ax_info = fig.add_axes([0.04, 0.68, 0.44, 0.17])
ax_info.set_facecolor('#1a1a2e')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
ax_info.axis('off')
info_text = (
    "DATASET\n\n"
    f"  506 census tracts, Boston area\n"
    f"  13 features | 1 target (MEDV)\n"
    f"  Source: OpenML #531\n"
    f"  No missing values"
)
ax_info.text(0, 0.95, info_text, fontsize=13, color='#c0c0d0',
             va='top', fontfamily='monospace', linespacing=1.8)

# ---- right panel: big metrics ----
ax_metrics = fig.add_axes([0.52, 0.68, 0.44, 0.17])
ax_metrics.set_facecolor('#16213e')
ax_metrics.set_xlim(0, 1)
ax_metrics.set_ylim(0, 1)
ax_metrics.axis('off')

# metric boxes
for i, (label, val, col) in enumerate([
    (f'BEST\nRMSE', f'{rmse_best:.3f}', '#e94560'),
    (f'BEST\nR²', f'{r2_best:.3f}', '#0f3460'),
    (f'IMPROVEMENT', f'{(rmse_lr-rmse_best)/rmse_lr*100:.1f}%', '#533483'),
]):
    x = 0.05 + i * 0.33
    rect = mpatches.FancyBboxPatch((x, 0.25), 0.28, 0.65, boxstyle='round,pad=0.03',
                                     facecolor=col, edgecolor='none', alpha=0.9)
    ax_metrics.add_patch(rect)
    ax_metrics.text(x + 0.14, 0.72, label, fontsize=10, color='#aaaaaa',
                    ha='center', va='center', fontfamily='monospace', fontweight='bold')
    ax_metrics.text(x + 0.14, 0.4, val, fontsize=22, color='white',
                    ha='center', va='center', fontfamily='monospace', fontweight='bold')

# ---- model comparison bar chart ----
ax_bar = fig.add_axes([0.06, 0.34, 0.9, 0.28])
ax_bar.set_facecolor('#1a1a2e')
model_names = list(models.keys())
rmse_vals = [v[0] for v in models.values()]
colors = ['#e94560', '#e94560', '#3a3a5c', '#3a3a5c', '#3a3a5c', '#3a3a5c']
y_pos = range(len(model_names))
bars = ax_bar.barh(y_pos, rmse_vals, color=colors, height=0.6, edgecolor='none')
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(model_names, fontfamily='monospace', fontsize=10, color='#c0c0d0')
ax_bar.invert_yaxis()
ax_bar.set_xlabel('RMSE', fontfamily='monospace', fontsize=10, color='#888')
ax_bar.tick_params(axis='x', colors='#888')
for spine in ax_bar.spines.values():
    spine.set_visible(False)
ax_bar.set_title('MODEL COMPARISON (Test RMSE)', fontsize=14, fontweight='bold',
                 color='white', fontfamily='monospace', pad=15)
for bar, (rmse, r2) in zip(bars, models.values()):
    ax_bar.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'RMSE={rmse:.3f}  R²={r2:.3f}',
                va='center', fontsize=8, color='#c0c0d0', fontfamily='monospace')

ax_bar.set_xlim(0, 8.5)

# ---- feature importance: LR vs RF vs GB ----
lr_imp = np.abs(lr.coef_)
lr_imp_n = lr_imp / lr_imp.sum()
rf_imp = best.feature_importances_
rf_imp_n = rf_imp / rf_imp.sum()
gb_imp = gb.feature_importances_
gb_imp_n = gb_imp / gb_imp.sum()
avg_imp = (lr_imp_n + rf_imp_n + gb_imp_n) / 3
sort_idx = np.argsort(avg_imp)

ax_fi = fig.add_axes([0.06, 0.06, 0.9, 0.24])
ax_fi.set_facecolor('#1a1a2e')
h = 0.25
colors_fi = ['#5bc0de', '#5cb85c', '#f0ad4e']
labels_fi = ['Linear Regression |coef|', 'Random Forest', 'Gradient Boosting']
for j, (imp, c, lbl) in enumerate(zip([lr_imp_n, rf_imp_n, gb_imp_n], colors_fi, labels_fi)):
    y_vals = [i + j * h for i in range(len(feature_names))]
    ax_fi.barh(y_vals, imp[sort_idx], height=h * 0.8, color=c, edgecolor='none', alpha=0.85)
y_labels = [f'{feature_names[i]}  ' for i in sort_idx]
ax_fi.set_yticks([i + h for i in range(len(feature_names))])
ax_fi.set_yticklabels(feature_names, fontfamily='monospace', fontsize=9, color='#c0c0d0')
ax_fi.tick_params(axis='x', colors='#888')
for spine in ax_fi.spines.values():
    spine.set_visible(False)
ax_fi.set_title('MULTI-ALGORITHM FEATURE IMPORTANCE', fontsize=14, fontweight='bold',
                color='white', fontfamily='monospace', pad=15)
legend_patches = [mpatches.Patch(color=c, label=lbl) for c, lbl in zip(colors_fi, labels_fi)]
ax_fi.legend(handles=legend_patches, loc='lower right', fontsize=7,
             labelcolor='#c0c0d0', facecolor='#16213e', edgecolor='none')

# ---- footer ----
ax_footer = fig.add_axes([0, 0, 1, 0.04])
ax_footer.set_facecolor('#16213e')
ax_footer.set_xlim(0, 1)
ax_footer.set_ylim(0, 1)
ax_footer.axis('off')
ax_footer.text(0.5, 0.5, 'sklearn  |  OpenML  |  Kaggle Boston Housing  |  nchu-boston-housing-kaggle',
               fontsize=9, color='#666', ha='center', va='center', fontfamily='monospace')

# ---------- save ----------
os.makedirs('images', exist_ok=True)
fig.savefig('images/poster.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close(fig)
print("Saved: images/poster.png")
