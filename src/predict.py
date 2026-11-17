"""
predict.py
Inference module for Food Delivery Time Prediction.
Supports single sample prediction (Distance, Traffic, Restaurant/City)
and batch prediction on test datasets (e.g., test.csv).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_and_preprocess


class DeliveryTimePredictor:
    def __init__(self, model_path="models/rf_delivery_model.pkl", features_path="models/feature_columns.pkl"):
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Model or feature definition not found at '{model_path}'. "
                "Please run `python src/train.py` first to train and save the model."
            )
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(features_path)

    def predict_custom(
        self,
        distance_km,
        traffic="Medium",
        city="Urban",
        order_type="Meal",
        vehicle_type="motorcycle",
        delivery_person_age=28.0,
        delivery_person_ratings=4.7,
        vehicle_condition=1,
        multiple_deliveries=0,
        festival="No",
        weather="Sunny",
    ):
        """
        Predict delivery time (in minutes) for specific inputs.
        Primary features: Distance, Traffic, Restaurant/City.
        """
        traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
        traffic_level = traffic_map.get(str(traffic).strip().title(), 2)
        festival_flag = 1 if str(festival).strip().lower() in ["yes", "1", "true"] else 0

        # Construct single-row DataFrame
        raw_input = {
            "distance_km": float(distance_km),
            "traffic_density_level": int(traffic_level),
            "Delivery_person_Age": float(delivery_person_age),
            "Delivery_person_Ratings": float(delivery_person_ratings),
            "Vehicle_condition": int(vehicle_condition),
            "multiple_deliveries": int(multiple_deliveries),
            "Festival": festival_flag,
            "City": str(city).strip(),
            "Type_of_order": str(order_type).strip(),
            "Type_of_vehicle": str(vehicle_type).strip(),
            "weather": str(weather).strip(),
        }

        df_single = pd.DataFrame([raw_input])
        df_encoded = pd.get_dummies(
            df_single,
            columns=["City", "Type_of_order", "Type_of_vehicle", "weather"],
            drop_first=True,
            dtype=float,
        )

        # Align with model's expected feature columns
        df_aligned = pd.DataFrame(0.0, index=[0], columns=self.feature_columns)
        for col in df_encoded.columns:
            if col in df_aligned.columns:
                df_aligned[col] = df_encoded[col]

        predicted_minutes = float(self.model.predict(df_aligned)[0])
        return round(predicted_minutes, 1)

    def predict_test_csv(self, test_csv_path="test.csv", output_csv_path="predictions_submission.csv"):
        """
        Runs batch prediction on a test CSV file and exports formatted output.
        """
        print(f"Reading test file: {test_csv_path}")
        df_test = pd.read_csv(test_csv_path)
        id_series = df_test["ID"].copy()

        X_test_prep, _ = clean_and_preprocess(test_csv_path, is_train=False)

        # Align columns
        X_aligned = pd.DataFrame(0.0, index=X_test_prep.index, columns=self.feature_columns)
        for col in X_test_prep.columns:
            if col in X_aligned.columns:
                X_aligned[col] = X_test_prep[col]

        preds = self.model.predict(X_aligned)
        output_df = pd.DataFrame({
            "ID": id_series,
            "Time_taken (min)": np.round(preds, 2),
        })
        output_df.to_csv(output_csv_path, index=False)
        print(f"Saved {len(output_df)} predictions to: {output_csv_path}")
        return output_df


if __name__ == "__main__":
    predictor = DeliveryTimePredictor()

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        test_file = sys.argv[2] if len(sys.argv) > 2 else "test.csv"
        predictor.predict_test_csv(test_csv_path=test_file)
    else:
        # Example demonstration predictions
        print("\n=== Sample Food Delivery Predictions ===")
        scenarios = [
            {"distance_km": 3.5, "traffic": "Low", "city": "Urban", "desc": "Short distance, Low traffic"},
            {"distance_km": 3.5, "traffic": "Jam", "city": "Urban", "desc": "Short distance, Traffic Jam"},
            {"distance_km": 12.0, "traffic": "Medium", "city": "Metropolitian", "desc": "Long distance, Medium traffic"},
            {"distance_km": 15.0, "traffic": "Jam", "city": "Metropolitian", "desc": "Long distance, Traffic Jam"},
        ]
        for s in scenarios:
            pred = predictor.predict_custom(
                distance_km=s["distance_km"],
                traffic=s["traffic"],
                city=s["city"],
            )
            print(f"- Scenario: {s['desc']:<35} -> Predicted Time: {pred:.1f} mins")

