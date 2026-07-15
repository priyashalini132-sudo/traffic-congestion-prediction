# Traffic Congestion Prediction & Smart Traffic Analytics System

> **End-to-end machine learning project** for predicting hourly traffic volume and real-time congestion risk scoring across road junctions.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Get the Dataset
Download **traffic.csv** from Kaggle:
👉 [Traffic Prediction Dataset — fedesoriano](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)

Place it at:
```
traffic-congestion-prediction/
└── data/
    └── raw/
        └── traffic.csv   ← here
```

The app will also attempt to auto-download the dataset at startup. If that fails, follow the manual step above.

### 3. Train the Models
```bash
python -m src.train
```
This will:
- Clean & preprocess the data
- Engineer temporal, cyclical, and historical features
- Train 5 ML models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- Auto-select the best model by R² score
- Save `models/best_model.pkl`, `models/preprocessor.pkl`, `models/model_metadata.json`
- Generate EDA figures in `reports/figures/`
- Write `reports/traffic_insights.md`

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 📁 Project Structure

```
traffic-congestion-prediction/
│
├── app.py                    ← Streamlit dashboard
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── .gitignore
├── runtime.txt               ← Python version
│
├── data/
│   ├── raw/
│   │   └── traffic.csv       ← Place dataset here
│   └── processed/
│       ├── cleaned_traffic.csv
│       ├── train_featured.csv
│       └── test_featured.csv
│
├── models/
│   ├── best_model.pkl        ← Saved best model
│   ├── preprocessor.pkl      ← Scaler + profiles
│   └── model_metadata.json   ← Metrics + config
│
├── notebooks/
│   └── traffic_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← Load & validate data
│   ├── preprocessing.py      ← Clean, deduplicate, interpolate
│   ├── feature_engineering.py← Temporal, cyclical, historical features
│   ├── train.py              ← Multi-model training pipeline
│   ├── evaluate.py           ← MAE, RMSE, R² metrics
│   ├── predict.py            ← Inference engine
│   ├── risk_scoring.py       ← 0-100 risk score + categorization
│   └── visualization.py      ← Plotly & Matplotlib charts
│
├── reports/
│   ├── figures/              ← Auto-generated EDA charts
│   ├── model_comparison.csv  ← All model metrics
│   └── traffic_insights.md   ← Auto-generated insights report
│
└── tests/
    └── test_pipeline.py      ← pytest test suite
```

---

## 🗂️ Dataset Schema

| Column     | Type     | Description                              |
|------------|----------|------------------------------------------|
| `DateTime` | datetime | Hourly timestamp of measurement          |
| `Junction` | int      | Junction ID (1–4)                        |
| `Vehicles` | int      | Number of vehicles recorded at that hour |
| `ID`       | int      | Unique record identifier                 |

---

## 🤖 ML Models Compared

| Model              | Notes                                      |
|--------------------|--------------------------------------------|
| Linear Regression  | Baseline linear model                      |
| Decision Tree      | Non-linear, max_depth=10                   |
| Random Forest      | 50 estimators, ensemble method             |
| Gradient Boosting  | 100 estimators, sequential boosting        |
| **XGBoost**        | **Typically best performer**               |

---

## 🔮 Feature Engineering

- **Temporal**: Hour, Day, DayOfWeek, Month, Year
- **Flags**: IsWeekend, IsRushHour, TimeOfDay
- **Cyclical**: sin/cos encoding for Hour, DayOfWeek, Month
- **Historical Profiles**: Junction × Hour × DayOfWeek average traffic (fitted only on train set — no leakage)

---

## 📊 Dashboard Pages

| Page               | Description                                              |
|--------------------|----------------------------------------------------------|
| 🏠 Overview         | KPIs, congestion heatmap, hourly trends, model summary   |
| 🔮 Live Predict     | Real-time prediction for any date, time & junction       |
| 📊 EDA & Analysis   | 5-tab interactive exploratory analysis                   |
| 🤖 Model Performance| Model comparison table, feature importances              |
| ℹ️ About            | Architecture, quick start, feature list                  |

---

## 📈 Congestion Risk Scoring

Risk scores are junction-specific, computed using training set quantiles:

| Level  | Score Range | Threshold          |
|--------|-------------|--------------------|
| 🟢 Low  | 0–33        | Below 33rd percentile |
| 🟡 Medium | 33–66    | 33rd–66th percentile  |
| 🔴 High  | 66–100    | Above 66th percentile |

---

## 🔧 Configuration

All paths are auto-resolved relative to the project root. No manual path configuration needed.

---

## 📄 License

MIT License — free to use, modify, and distribute.
