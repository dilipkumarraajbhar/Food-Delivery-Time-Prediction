"""
train.py
Model training script for Food Delivery Time Prediction using Random Forest Regressor.
"""

import json
import os
import sys

# Ensure clean matplotlib cache on macOS
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_and_preprocess


def train_model(
    data_path="train.csv",
    models_dir="models",
    plots_dir="plots",
    n_estimators=100,
    max_depth=16,
    random_state=42,
):
    print("=" * 60)
    print("  Food Delivery Time Prediction - Model Training Pipeline  ")
    print("=" * 60)

    # 1. Load and preprocess data
    print(f"\n[1/5] Loading and preprocessing dataset from '{data_path}'...")
    X, y = clean_and_preprocess(data_path, is_train=True)
    print(f"      Total samples: {X.shape[0]}")
    print(f"      Total features: {X.shape[1]}")
    print(f"      Features list: {list(X.columns)}")

    # 2. Train-Test Split (80% train, 20% test)
    print("\n[2/5] Splitting dataset into 80% Train and 20% Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Testing set:  {X_test.shape[0]} samples")

    # 3. Model Training
    print("\n[3/5] Training Random Forest Regressor...")
    print(
        f"      Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}"
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("      Model training complete!")

    # 4. Evaluation
    print("\n[4/5] Evaluating model performance on Test Set...")
    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "-" * 40)
    print("        MODEL EVALUATION RESULTS        ")
    print("-" * 40)
    print(f"  Mean Absolute Error (MAE):    {mae:.2f} minutes")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
    print(f"  R-squared (R2 Score):          {r2:.4f} ({r2*100:.1f}% variance explained)")
    print("-" * 40)

    # 5. Save Artifacts & Visualizations
    print("\n[5/5] Saving model and generating evaluation plots...")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save model and feature names
    model_path = os.path.join(models_dir, "rf_delivery_model.pkl")
    features_path = os.path.join(models_dir, "feature_columns.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(rf, model_path)
    joblib.dump(list(X.columns), features_path)

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2_Score": round(float(r2), 4),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": list(X.columns),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"      Saved model to: {model_path}")
    print(f"      Saved feature names to: {features_path}")
    print(f"      Saved metrics to: {metrics_path}")

    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [X.columns[i] for i in indices]
    sorted_importances = importances[indices]

    plt.barh(sorted_features[:10][::-1], sorted_importances[:10][::-1], color="#2E86AB")
    plt.xlabel("Relative Feature Importance")
    plt.title("Top 10 Feature Importances in Food Delivery Time Prediction")
    plt.tight_layout()
    fi_plot_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=200)
    plt.close()
    print(f"      Saved feature importance plot to: {fi_plot_path}")

    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.15, color="#06D6A0", edgecolors="none")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    plt.xlabel("Actual Delivery Time (minutes)")
    plt.ylabel("Predicted Delivery Time (minutes)")
    plt.title(f"Actual vs. Predicted Delivery Time (R² = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    avp_plot_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    plt.savefig(avp_plot_path, dpi=200)
    plt.close()
    print(f"      Saved actual vs predicted plot to: {avp_plot_path}")

    # Plot 3: Residuals Distribution
    plt.figure(figsize=(8, 5))
    residuals = y_test - y_pred
    plt.hist(residuals, bins=40, color="#118AB2", edgecolor="white", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Residual (Actual - Predicted in min)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Error Distribution (MAE = {mae:.2f} min)")
    plt.tight_layout()
    res_plot_path = os.path.join(plots_dir, "residual_distribution.png")
    plt.savefig(res_plot_path, dpi=200)
    plt.close()
    print(f"      Saved residual plot to: {res_plot_path}")

    print("\nTraining completed successfully!")
    return metrics


if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    train_model(data_path=data_file)

