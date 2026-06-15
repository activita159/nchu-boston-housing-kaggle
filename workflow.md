# 房價預測 — 工作流程

## 快速開始

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python housing_pipeline.py --config config/boston.json
```

執行後，所有圖表儲存於 `{output.dir}/images/`，預測結果儲存於 `submission.csv`。

---

## 流程步驟

### 步驟 1 — 載入資料

支援兩種資料來源：

| 來源 | 設定鍵值 | 範例 |
|------|----------|------|
| OpenML | `data.source: "openml"` | Boston Housing (`openml_name: "boston"`) |
| CSV | `data.source: "csv"` | 本地檔案 (`csv_path: "data/tokyo.csv"`) |

### 步驟 2 — 探索性資料分析（EDA）與視覺化

| 輸出檔案 | 說明 |
|----------|------|
| `correlation_heatmap.png` | 所有變數的上三角相關性熱力圖 |
| `target_distribution.png` | 目標變數直方圖 + 箱形圖 |
| `feature_scatter.png` | 與目標變數最相關的前 N 個特徵散佈圖（附 Pearson r） |

終端輸出：`df.describe()`、缺失值統計。

### 步驟 3 — 資料預處理

- 訓練/測試集分割（預設 80/20，`random_state` 取自設定檔）
- `StandardScaler` 在訓練集上擬合，兩組皆進行轉換

### 步驟 4 — 線性回歸基準模型

| 指標 | 數值 |
|------|------|
| RMSE | 依資料集而異 |
| MAE | 依資料集而異 |
| R² | 依資料集而異 |

| 輸出檔案 | 說明 |
|----------|------|
| `linear_coefficients.png` | 標準化係數（水平長條圖，正/負以不同顏色標示） |
| `residuals.png` | 殘差 vs 擬合值 + 殘差分佈直方圖 |
| `predicted_vs_actual.png` | 預測 vs 實際散佈圖，附 45 度參考線 |
| `linear_regression_fit.png` | 個別特徵回歸擬合線（取自 `visualization.fit_line_features` 設定） |

### 步驟 5 — 多模型比較

六種模型在測試集上比較，搭配 K 折交叉驗證：

| 模型 | 測試 RMSE | 測試 R² | CV RMSE（平均） |
|------|-----------|---------|-----------------|
| Gradient Boosting | | | |
| Random Forest | | | |
| Linear Regression | | | |
| Ridge | | | |
| ElasticNet | | | |
| Lasso | | | |

### 步驟 5b — 多演算法特徵重要性比較

| 輸出檔案 | 說明 |
|----------|------|
| `multi_algo_importance.png` | 三欄比較：線性回歸 (|coef|)、Random Forest、Gradient Boosting |

### 步驟 6 — 超參數調優

使用 `GridSearchCV` 對 `RandomForestRegressor` 進行調參，參數範圍取自 `config.models.rf_tune`。

| 參數 | 預設搜尋範圍 | 最佳值 |
|------|-------------|--------|
| `n_estimators` | {100, 200, 300} | 依資料集而異 |
| `max_depth` | {5, 10, 15, None} | 依資料集而異 |
| `min_samples_split` | {2, 5} | 依資料集而異 |
| `min_samples_leaf` | {1, 2} | 依資料集而異 |

### 步驟 7 — 最佳模型視覺化

| 輸出檔案 | 說明 |
|----------|------|
| `best_model_predicted_vs_actual.png` | 調優後 RF 預測 vs 實際散佈圖 |
| `rf_feature_importance.png` | 調優後 RF 特徵重要性（水平長條圖） |

### 步驟 8 — 提交檔案

| 輸出檔案 | 說明 |
|----------|------|
| `submission.csv` | 測試集預測結果 |

---

## 檔案結構

```
project/
├── housing_pipeline.py            # 主管線（讀取設定檔）
├── generate_poster.py             # 海報生成器（讀取設定檔）
├── config/
│   ├── boston.json                # Boston 房價設定檔
│   └── tokyo.json                 # 其他城市設定檔（使用者自行建立）
├── output/
│   ├── boston/
│   │   ├── images/
│   │   └── submission.csv
│   └── tokyo/
│       ├── images/
│       └── submission.csv
├── README.md
├── report.md
└── workflow.md
```
