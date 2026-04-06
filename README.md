# 📈 Multi-Asset Financial Forecaster

ML-powered 6-month price predictions for **Gold, Silver, Reliance Industries & HDFC Bank** — all in **Indian Rupees (₹)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![ML](https://img.shields.io/badge/ML-scikit--learn-orange)

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🏗 App Structure

```
financial-forecaster/
├── app.py                  # Flask app + ML pipeline (main entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Saved trained models (.pkl)
├── data/                   # Cached prediction results (.json)
├── static/
│   ├── css/
│   │   └── style.css       # Dark-themed professional UI
│   ├── js/
│   │   └── app.js          # Frontend logic + Plotly charts
│   └── images/             # (optional assets)
├── templates/
│   └── index.html          # Main dashboard template
└── utils/                  # (extensible: add custom modules)
```

---

## 📊 Assets Covered

| Asset | Ticker | Source Currency | Display |
|-------|--------|---------------|---------|
| Gold | GC=F | USD → INR | ₹/oz |
| Silver | SI=F | USD → INR | ₹/oz |
| Reliance Industries | RELIANCE.NS | INR | ₹ |
| HDFC Bank | HDFCBANK.NS | INR | ₹ |

---

## 🤖 ML Models

Three models are trained per asset and the best (by RMSE) is selected:

1. **Linear Regression** — baseline
2. **Random Forest** (200 trees, depth 15) — ensemble
3. **Gradient Boosting** (200 estimators, lr=0.05) — sequential ensemble

### Features Engineered
- Moving averages (5, 10, 20, 50 days)
- Rolling standard deviation
- Percentage changes (1d, 5d)
- Lag features (1, 5, 20 days)
- Calendar features (day of year, month, weekday)

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

Displayed in the UI after each prediction.

---

## 📡 Data Source

Uses **yfinance** (Yahoo Finance API) — no API key needed.

USD → INR conversion fetched live via `INR=X` ticker.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Predictions are based on historical patterns and do not constitute financial advice. Past performance does not guarantee future results.
