# 房價預測 — 通用機器學習pipeline

一個可重複使用的房價預測機器學習pipeline，透過更換 JSON 設定檔即可套用到不同城市/地區。

## 功能特點

- **設定檔驅動**：所有資料集相關參數集中在 `config/*.json`
- **多城市適用**：同一pipeline可跑 Boston、Manhattan 或任何房價資料集
- **完整流程**：EDA、預處理、6 種模型比較、GridSearchCV 調參、Kaggle 提交
- **豐富視覺化**：每次執行產出 10+ 張圖表（相關性熱力圖、殘差、特徵重要性等）
- **海報生成**：學術會議風格摘要海報

## 快速開始

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# 用任意設定檔執行
python housing_pipeline.py --config config/boston.json

# 生成海報
python generate_poster.py --config config/boston.json
```

## 專案結構

```
.
├── housing_pipeline.py            # 通用pipeline（讀取設定檔）
├── generate_poster.py             # 通用海報生成器（讀取設定檔）
├── config/
│   ├── boston.json                # Boston 房價設定檔
│   └── manhattan.json             # Manhattan 房價設定檔（自行建立）
├── output/
│   └── boston/
│       ├── images/                # 所有產出圖表
│       └── submission.csv         # Kaggle 格式預測結果
├── report.md                      # 分析報告範本
└── workflow.md                    # 詳細流程說明
```

## 如何新增一個城市

### 1. 建立設定檔

複製並修改設定檔：

```bash
cp config/manhattan.json config/tokyo.json
```

### 2. 編輯設定檔內容

```json
{
  "project_name": "Tokyo 房價",
  "project_slug": "tokyo-housing",
  "data": {
    "source": "csv",
    "csv_path": "data/tokyo.csv",
    "target_column": "price",
    "target_display": "Price",
    "target_desc": "東京住宅中位數價格",
    "target_unit": "¥",
    "dataset_source": "Tokyo Open Data",
    "dataset_desc": "東京住宅不動產交易資料"
  },
  "output": {
    "dir": "output/tokyo",
    "submission_target_col": "price"
  }
}
```

### 3. 執行

```bash
python housing_pipeline.py --config config/tokyo.json
python generate_poster.py --config config/tokyo.json
```

## 設定檔欄位說明

| 欄位 | 必填 | 說明 |
|-------|------|------|
| `project_name` | 是 | 圖表與輸出使用的專案名稱 |
| `project_slug` | 是 | 用於路徑的簡短識別名稱 |
| `data.source` | 是 | `"openml"` 或 `"csv"` |
| `data.target_column` | 是 | 目標變數的欄位名稱 |
| `data.openml_name` | source=openml 時 | OpenML 資料集名稱 |
| `data.csv_path` | source=csv 時 | CSV 檔案路徑 |
| `data.target_display` | 否 | 圖表中目標變數的顯示名稱（預設用 target_column） |
| `data.target_desc` | 否 | 終端輸出用的描述文字 |
| `data.target_unit` | 否 | 單位字串（例如 "$1000s"） |
| `visualization.fit_line_features` | 否 | 擬合線圖要畫的特徵清單（預設取相關性最高的前 2 個） |
| `training.test_size` | 否 | 測試集分割比例（預設 0.2） |
| `training.random_state` | 否 | 隨機種子（預設 42） |
| `models.rf_tune` | 否 | Random Forest GridSearchCV 參數範圍 |
| `output.dir` | 是 | 本次執行的輸出目錄 |

## 兩種資料來源

| 來源 | 設定方式 | 適用情境 |
|------|----------|----------|
| **OpenML** | `"source": "openml"` + 指定 `openml_name` | 標準公開資料集（如 Boston） |
| **本地 CSV** | `"source": "csv"` + 指定 `csv_path` | 自己收集的資料 |

**CSV 格式要求**：第一行為欄位名稱，數值型特徵欄位 + 一個目標變數欄位（不需要 ID 欄），欄位順序不拘。

## 執行結果（Boston Housing）

| 模型 | RMSE | R² |
|-------|------|-----|
| Gradient Boosting | 2.445 | 0.918 |
| Random Forest | 2.915 | 0.884 |
| Ridge | 4.931 | 0.668 |
| Linear Regression | 4.929 | 0.669 |
| ElasticNet | 5.020 | 0.656 |
| Lasso | 5.065 | 0.650 |

**最佳模型**：調參後的 Random Forest — Test RMSE 3.024、R² 0.875，相較基準模型改善 38.64%。
