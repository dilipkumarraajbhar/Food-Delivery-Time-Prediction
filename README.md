# 🍔 Food Delivery Time Prediction

### 👨‍💻 Student Details
- **Name**: Dilip
- **UID**: 26BCS12620
- **Section**: 508 A

---

### 📋 Project Overview
This project develops an end-to-end Machine Learning regression solution that predicts food delivery duration (in minutes). It models the complex interaction between:
- **Geodesic Distance** calculated from GPS coordinates via the Haversine formula
- **Live Traffic Density** (Low, Medium, High, Jam)
- **Environmental & Partner Factors** (Weather, Vehicle type, Delivery partner ratings, and order prep delay)

Trained and evaluated on real-world Kaggle delivery records using an optimized **Random Forest Regressor**.

- **Problem Type**: Regression
- **Dataset**: Kaggle – Food Delivery Dataset (`train.csv`, `test.csv`)
- **Objective**: Accurately predict delivery time in minutes.
- **ML Technique**: Random Forest Regressor

---

## 📊 Model Performance Summary

| Metric | Score | Description |
|---|---|---|
| **R² Score** | **0.8066** | Explains **80.7%** of the variance in delivery times |
| **Mean Absolute Error (MAE)** | **3.26 min** | Average prediction error is only ~3.2 minutes |
| **Root Mean Squared Error (RMSE)** | **4.12 min** | Penalizes larger errors |

---

## 🌐 Interactive Web Application

A full-fledged web interface built with **Streamlit** is included. It provides:
- Live sliders for **Delivery Distance** and dropdown for **Traffic Density**.
- City type, order type, partner ratings, and weather inputs.
- Instant estimated delivery time cards with express/standard/delayed status badges.
- Model performance plots and methodological architecture.

To run the web app:
```bash
./venv/bin/streamlit run app.py
```

---

## 📁 Project Structure

```
FDTP_Project/
│── app.py                          # Streamlit Interactive Web Application
│── demo.py                         # Interactive CLI prediction demo
│── train.csv                       # Training dataset with target Time_taken(min)
│── test.csv                        # Test dataset for predictions
│── Sample_Submission.csv           # Kaggle sample submission format
│── Food_Delivery_Time_Prediction.ipynb # Complete Jupyter Notebook for submission
│── requirements.txt                # Python dependencies
│── src/
│   ├── preprocess.py               # Haversine distance, traffic ordinality, cleaning
│   ├── train.py                    # Model training, hyperparameter tuning & evaluation
│   └── predict.py                  # Single sample & batch test predictions
│── models/
│   ├── rf_delivery_model.pkl       # Trained Random Forest model
│   ├── feature_columns.pkl         # Serialized feature column names
│   └── metrics.json                # Model performance metrics
└── plots/
    ├── feature_importance.png      # Top feature importance bar chart
    ├── actual_vs_predicted.png     # Actual vs Predicted scatter plot
    └── residual_distribution.png   # Error residual histogram
```

---

## 🚀 How to Run

### 1. Launch the Web App
```bash
./venv/bin/streamlit run app.py
```
*(Automatically opens at `http://localhost:8501` in your browser)*

### 2. Run the Command-Line Demo
```bash
./venv/bin/python3 demo.py
```

### 3. Re-train the Model
```bash
./venv/bin/python3 src/train.py
```

### 4. Run Batch Predictions on `test.csv`
```bash
./venv/bin/python3 src/predict.py --batch test.csv
```

### 5. Jupyter Notebook
Open `Food_Delivery_Time_Prediction.ipynb` in VS Code or JupyterLab for the academic submission report.
