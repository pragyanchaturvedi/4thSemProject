"""
Multi-Asset Financial Forecaster
Flask application for predicting Gold, Silver, Reliance & HDFC prices in INR.
Single-command startup: python app.py
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import yfinance as yf

app = Flask(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ASSETS = {
    "Gold": {"ticker": "GC=F", "currency": "USD", "name": "Gold (INR/oz)"},
    "Silver": {"ticker": "SI=F", "currency": "USD", "name": "Silver (INR/oz)"},
    "Reliance": {"ticker": "RELIANCE.NS", "currency": "INR", "name": "Reliance Industries"},
    "HDFC": {"ticker": "HDFCBANK.NS", "currency": "INR", "name": "HDFC Bank"},
}

FORECAST_DAYS = 180  # ~6 months
LOOKBACK_YEARS = 5
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def get_usd_inr_rate():
    """Fetch current USD/INR exchange rate."""
    try:
        fx = yf.Ticker("INR=X")
        rate = fx.history(period="5d")["Close"].dropna().iloc[-1]
        return float(rate)
    except Exception:
        return 83.5  # fallback


def fetch_asset_data(ticker, years=LOOKBACK_YEARS):
    """Download historical daily close prices."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df[["Close"]].dropna().copy()
    df.columns = ["close"]
    df.index = pd.to_datetime(df.index)
    # Flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def create_features(df):
    """Engineer time-series features from price data."""
    df = df.copy()
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["day_of_week"] = df.index.dayofweek

    for w in [5, 10, 20, 50]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"std_{w}"] = df["close"].rolling(w).std()

    df["pct_change_1"] = df["close"].pct_change(1)
    df["pct_change_5"] = df["close"].pct_change(5)
    df["lag_1"] = df["close"].shift(1)
    df["lag_5"] = df["close"].shift(5)
    df["lag_20"] = df["close"].shift(20)

    df.dropna(inplace=True)
    return df


def train_and_predict(asset_key):
    """Train ensemble models and generate 6-month forecast."""
    cfg = ASSETS[asset_key]
    usd_inr = get_usd_inr_rate() if cfg["currency"] == "USD" else 1.0

    # Fetch & prepare data
    raw_df = fetch_asset_data(cfg["ticker"])
    raw_df["close"] = raw_df["close"] * usd_inr  # convert to INR
    df = create_features(raw_df)

    feature_cols = [c for c in df.columns if c != "close"]
    X = df[feature_cols].values
    y = df["close"].values

    # Train/test split (last 60 days = test)
    split = -60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train_s)
        pred_s = model.predict(X_test_s)
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "test_pred": pred}

    # Pick best model by RMSE
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]

    # Save model
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{asset_key}_model.pkl"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, f"{asset_key}_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, f"{asset_key}_scaler_y.pkl"))

    # ── Generate 6-month forecast ──
    last_row = df.iloc[-1:].copy()
    forecast_dates = []
    forecast_prices = []
    current_close = float(df["close"].iloc[-1])

    for i in range(FORECAST_DAYS):
        future_date = df.index[-1] + timedelta(days=i + 1)
        if future_date.weekday() >= 5:
            continue
        row = last_row.copy()
        row.index = [future_date]
        row["day_of_year"] = future_date.timetuple().tm_yday
        row["month"] = future_date.month
        row["year"] = future_date.year
        row["day_of_week"] = future_date.weekday()
        row["lag_1"] = current_close
        feat = row[feature_cols].values
        feat_s = scaler_X.transform(feat)
        pred_s = best_model.predict(feat_s)
        pred_price = float(scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()[0])
        current_close = pred_price
        forecast_dates.append(future_date.strftime("%Y-%m-%d"))
        forecast_prices.append(round(pred_price, 2))
        # Update rolling features approximately
        last_row = row.copy()
        last_row["close"] = pred_price

    # Historical data for charts
    hist = raw_df.tail(252)  # last ~1 year
    hist_dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    hist_prices = [round(float(p), 2) for p in hist["close"].values]

    return {
        "asset": asset_key,
        "name": cfg["name"],
        "currency": "INR",
        "best_model": best_name,
        "metrics": {name: {"rmse": round(r["rmse"], 2), "mae": round(r["mae"], 2)} for name, r in results.items()},
        "current_price": round(float(df["close"].iloc[-1]), 2),
        "historical": {"dates": hist_dates, "prices": hist_prices},
        "forecast": {"dates": forecast_dates, "prices": forecast_prices},
        "usd_inr_rate": round(usd_inr, 2),
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", assets=ASSETS)


@app.route("/api/predict/<asset_key>")
def predict(asset_key):
    if asset_key not in ASSETS:
        return jsonify({"error": "Unknown asset"}), 404
    try:
        result = train_and_predict(asset_key)
        # Cache result
        with open(os.path.join(DATA_DIR, f"{asset_key}_result.json"), "w") as f:
            json.dump(result, f)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets")
def list_assets():
    return jsonify(ASSETS)


if __name__ == "__main__":
    print("\n🚀 Multi-Asset Financial Forecaster")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
