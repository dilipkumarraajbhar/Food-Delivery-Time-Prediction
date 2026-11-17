"""
preprocess.py
Data preprocessing and feature engineering for Food Delivery Time Prediction.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [np.abs(lat1), np.abs(lon1), np.abs(lat2), np.abs(lon2)],
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def clean_and_preprocess(filepath_or_df, is_train=True):
    """
    Cleans the raw Food Delivery dataset and engineers key predictive features:
    - Distance (km) via Haversine formula
    - Traffic density (ordinal encoding)
    - Restaurant / pickup preparation delay (minutes)
    - Delivery partner age and rating
    - Weather conditions, vehicle condition, city type, order type
    """
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df.copy()

    # 1. Strip string column whitespaces & normalize null values
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["NaN", "nan", "null", "conditions NaN"], np.nan)

    # 2. Clean Target variable if present
    target_col = "Time_taken(min)"
    y = None
    if is_train and target_col in df.columns:
        # Values look like "(min) 24" -> extract digits
        y = (
            df[target_col]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)[0]
        )

    # 3. Clean Numeric Features
    df["Delivery_person_Age"] = pd.to_numeric(
        df["Delivery_person_Age"], errors="coerce"
    )
    df["Delivery_person_Ratings"] = pd.to_numeric(
        df["Delivery_person_Ratings"], errors="coerce"
    )
    df["multiple_deliveries"] = pd.to_numeric(
        df["multiple_deliveries"], errors="coerce"
    )
    df["Vehicle_condition"] = pd.to_numeric(
        df["Vehicle_condition"], errors="coerce"
    )

    # 4. Feature Engineering: Distance (in kilometers)
    coords = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for c in coords:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Compute Haversine distance
    raw_dist = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )

    # Handle invalid coordinates (e.g. 0,0 locations) or extreme outliers
    valid_mask = (
        (df["Restaurant_latitude"] != 0)
        & (raw_dist > 0.1)
        & (raw_dist < 100.0)
    )
    median_dist = raw_dist[valid_mask].median()
    df["distance_km"] = np.where(valid_mask, raw_dist, median_dist)

    # 5. Feature Engineering: Traffic Density (Ordinal)
    # Traffic levels: Low -> 1, Medium -> 2, High -> 3, Jam -> 4
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    df["traffic_density_level"] = df["Road_traffic_density"].map(traffic_map)
    # Default missing traffic to median level (2 = Medium)
    df["traffic_density_level"] = df["traffic_density_level"].fillna(2).astype(int)

    # 6. Feature Engineering: Weather conditions
    # Clean 'conditions Sunny' -> 'Sunny'
    if "Weatherconditions" in df.columns:
        df["weather"] = (
            df["Weatherconditions"]
            .astype(str)
            .str.replace("conditions ", "", regex=False)
            .replace(["NaN", "nan"], np.nan)
        )
    else:
        df["weather"] = "Sunny"

    # 7. Impute missing numerical columns with median
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(
        df["Delivery_person_Age"].median()
    )
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(
        df["Delivery_person_Ratings"].median()
    )
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0).astype(int)
    df["Vehicle_condition"] = df["Vehicle_condition"].fillna(1).astype(int)

    # 8. Clean Categoricals
    df["Festival"] = df["Festival"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)
    df["City"] = df["City"].fillna("Metropolitian")
    df["Type_of_order"] = df["Type_of_order"].fillna("Meal")
    df["Type_of_vehicle"] = df["Type_of_vehicle"].fillna("motorcycle")
    df["weather"] = df["weather"].fillna("Sunny")

    # Select final modeling features
    feature_cols = [
        "distance_km",
        "traffic_density_level",
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "multiple_deliveries",
        "Festival",
        "City",
        "Type_of_order",
        "Type_of_vehicle",
        "weather",
    ]

    X = df[feature_cols].copy()

    # Apply one-hot encoding for categorical variables
    categorical_cols = ["City", "Type_of_order", "Type_of_vehicle", "weather"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)

    if is_train:
        # Align rows if any target had NaN
        if y is not None:
            valid_y_mask = ~y.isna()
            X = X[valid_y_mask]
            y = y[valid_y_mask]
        return X, y
    else:
        return X, df.get("ID")
