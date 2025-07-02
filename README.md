
# 📈 Hybrid ARIMA-LSTM Stock Price Prediction

This project implements a **Hybrid ARIMA-LSTM model** to forecast stock prices, using NVIDIA (NVDA) as the target stock. It combines the strengths of both:

- **ARIMA**: for capturing linear trends and seasonality in the time series.
- **LSTM**: for modeling nonlinear patterns in the residuals that ARIMA fails to capture.

---

## 🚀 Objective

To build a robust stock forecasting model that:
- Detects underlying linear trends using ARIMA.
- Learns complex nonlinear behavior via LSTM on residuals.
- Combines both predictions to generate a final hybrid forecast.
- Achieves strong evaluation metrics (e.g., R² score).

---

## 🧠 Model Architecture

### 1. **Preprocessing**
- Fetch historical stock prices from Yahoo Finance via `yfinance`.
- Use Augmented Dickey-Fuller (ADF) test to check stationarity.
- Apply first-order differencing if the series is non-stationary.

### 2. **ARIMA Model**
- Manually set ARIMA `(p,d,q)` parameters (default: `(1,1,0)`).
- Fit on training data to predict linear components.
- Extract residuals: `residuals = actual - ARIMA prediction`.

### 3. **LSTM Model**
- Train LSTM on scaled residuals using sliding windows.
- Predict nonlinear residuals.
- Add LSTM predictions back to ARIMA output to generate hybrid forecast.

### 4. **Postprocessing & Evaluation**
- Reconstruct actual stock prices from differenced predictions using cumulative sum.
- Evaluate using:
  - **R² Score**
  - **Visual Plot of Predictions vs Actual**

---

## 📊 Example Results

| Metric         | Value     |
|----------------|-----------|
| R² Score       | ~0.98    |
| Dataset Used   | NVDA (2015–2024) |
| Data Source    | Yahoo Finance |

> Note: Results may vary based on ARIMA parameters and time step used in LSTM.

---

## 🛠️ Technologies Used

- Python 3.x
- NumPy, Pandas, Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Statsmodels (ARIMA, ADF)
- `yfinance` for data fetching

---

## 📁 File Structure

```
.
├── hybrid_arima_lstm.ipynb   # Jupyter Notebook with full implementation
├── README.md                 # This file
└── requirements.txt          # (optional) pip requirements
```

---

## ⚙️ How to Run

1. Install dependencies:
   ```bash
   pip install yfinance statsmodels scikit-learn tensorflow matplotlib
   ```

2. Run the notebook:
   - Open `hybrid_arima_lstm.ipynb` in Jupyter or Colab.
   - Follow the cells step-by-step.

---

## 🧪 Future Improvements

- Use `auto_arima` for automatic `(p,d,q)` tuning.
- Incorporate other financial indicators (MACD, RSI, Volume).
- Add support for multivariate time series.
- Add early stopping to LSTM.
- Try GRU/Bidirectional LSTM for residuals.

---

## 📬 Contact

Made by [Ashwina Yadav]  
🔗 GitHub: [github.com/ashwinayadav6281](https://github.com/ashwinayadav6281)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
