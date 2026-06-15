# Boston Housing — 工作流程摘要

## 快速開始

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python boston_housing.py
```

執行後會在 `images/` 產生所有圖表、`submission.csv`，並在終端輸出最終摘要。

---

## 流程步驟

### 步驟 1 — 取得資料

| 項目 | 詳細 |
|------|------|
| 來源 | `fetch_openml(name='boston', version=1, as_frame=True)` |
| 樣本數 | 506 |
| 特徵 | 13 個數值變數 (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT) |
| 目標 | MEDV（自住房屋中位數價格，單位：千美元） |
| 缺失值 | 無 |

### 步驟 2 — 探索性資料分析 (EDA) 與視覺化

| 輸出檔案 | 說明 |
|----------|------|
| `images/correlation_heatmap.png` | 14 個變數的上三角相關性熱力圖 |
| `images/target_distribution.png` | MEDV 直方圖 + 箱形圖 |
| `images/feature_scatter.png` | 2x2 散佈圖：與 MEDV 相關性最高的前 4 個特徵，附 Pearson r |

終端輸出：`df.describe()`、缺失值統計。

### 步驟 3 — 資料預處理

- 80/20 訓練/測試集分割（`random_state=42`，404 筆訓練 / 102 筆測試）
- `StandardScaler` 在訓練集上擬合，兩組皆進行轉換

### 步驟 4 — 線性回歸基準模型

| 指標 | 數值 |
|------|------|
| RMSE | 4.929 |
| MAE | 3.189 |
| R^2 | 0.669 |

| 輸出檔案 | 說明 |
|----------|------|
| `images/linear_coefficients.png` | 水平長條圖：標準化係數（正/負以不同顏色標示） |
| `images/residuals.png` | 左：殘差 vs 擬合值；右：殘差分佈直方圖 |
| `images/predicted_vs_actual.png` | 預測 vs 實際散佈圖，附 45 度參考線 |
| `images/linear_regression_fit.png` | RM vs MEDV 與 LSTAT vs MEDV，附回歸線 + 方程式 + R^2 |

### 步驟 5 — 多模型比較

訓練並評估六種模型：

| 模型 | 測試 RMSE | 測試 R^2 | CV RMSE（平均） |
|------|-----------|----------|-----------------|
| Gradient Boosting | 2.445 | 0.918 | 3.645 |
| Random Forest | 2.915 | 0.884 | 3.847 |
| Linear Regression | 4.929 | 0.669 | 4.829 |
| Ridge | 4.931 | 0.668 | 4.829 |
| ElasticNet | 5.020 | 0.656 | 4.871 |
| Lasso | 5.065 | 0.650 | 4.886 |

### 步驟 5b — 多演算法特徵重要性比較

| 輸出檔案 | 說明 |
|----------|------|
| `images/multi_algo_importance.png` | 三欄水平長條圖：線性回歸 (|coef|)、Random Forest、Gradient Boosting — 全部歸一化，按平均重要性排序，共用 Y 軸 |

### 步驟 6 — 超參數調優

對 `RandomForestRegressor` 進行 `GridSearchCV`，搭配 5 折交叉驗證：

| 參數 | 搜尋範圍 | 最佳值 |
|------|---------|--------|
| `n_estimators` | {100, 200, 300} | 100 |
| `max_depth` | {5, 10, 15, None} | 10 |
| `min_samples_split` | {2, 5} | 2 |
| `min_samples_leaf` | {1, 2} | 2 |

| 指標 | 數值 |
|------|------|
| 最佳 CV RMSE | 3.810 |
| 測試 RMSE | 3.024 |
| 測試 R^2 | 0.875 |
| 相較線性回歸改善 | +38.64% |

### 步驟 7 — 最佳模型視覺化

| 輸出檔案 | 說明 |
|----------|------|
| `images/best_model_predicted_vs_actual.png` | 調優後 RF：預測 vs 實際散佈圖 + 45 度線 |
| `images/rf_feature_importance.png` | 調優後 RF 特徵重要性（水平長條圖） |

### 步驟 8 — 提交檔案

| 輸出檔案 | 說明 |
|----------|------|
| `submission.csv` | Kaggle 格式：`ID`、`MEDV`（102 筆，最佳模型在測試集上的預測結果） |

---

## 檔案結構

```
nchu-boston-housing-kaggle/
├── boston_housing.py              # 主流程腳本（一鍵執行）
├── images/                        # 所有產出的圖表（10 張 PNG）
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│   ├── feature_scatter.png
│   ├── linear_coefficients.png
│   ├── residuals.png
│   ├── predicted_vs_actual.png
│   ├── linear_regression_fit.png
│   ├── multi_algo_importance.png
│   ├── best_model_predicted_vs_actual.png
│   └── rf_feature_importance.png
├── submission.csv                 # Kaggle 格式預測結果
├── README.md                      # 快速參考
├── report.md                      # 完整分析報告（內嵌圖片）
└── workflow.md                    # 本檔案
```

## 關鍵結論

1. **LSTAT** 與 **RM** 是所有模型中最主要的預測因子
2. 樹狀集成模型（Gradient Boosting、Random Forest）遠優於線性/正則化模型 — 資料中存在非線性關係
3. 正則化（Ridge/Lasso）對這個資料集並未帶來優於普通線性回歸的效益
4. MEDV 的上限截斷（$50,000）對所有模型都構成預測挑戰
5. 調優後的 Random Forest 達到 **R^2 = 0.875**，相較基準模型的 **RMSE 降低了 38.64%**
