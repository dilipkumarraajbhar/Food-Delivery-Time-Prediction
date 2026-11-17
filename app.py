"""
app.py
Interactive Streamlit Web Application for Food Delivery Time Prediction.
Project: 70. Food Delivery Time Prediction (Section-410 B)
ML Algorithm: Random Forest Regressor
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

# Add current path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.predict import DeliveryTimePredictor

# Page Configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E23744;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #FFF5F5 0%, #FFE3E3 100%);
        border-radius: 12px;
        padding: 24px;
        border-left: 6px solid #E23744;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
    .result-number {
        font-size: 3rem;
        font-weight: 800;
        color: #E23744;
    }
    .metric-chip {
        display: inline-block;
        background: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #333333;
        margin-right: 8px;
        margin-bottom: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    /* Hide Deploy button, 3-dots menu, and footer */
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stDeployButton"] {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    return DeliveryTimePredictor()


try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <div style="font-size: 2.8rem; line-height: 1;">🛵💨</div>
            <h2 style="margin: 8px 0 2px 0; font-size: 1.4rem; font-weight: 700;">Food Delivery ETA</h2>
            <span style="background-color: #E23744; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                ML REGRESSION SYSTEM
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Student Details Card
    st.markdown("### 👨‍💻 Student Details")
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.05); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 12px;">
            <div style="font-size: 0.95rem; margin-bottom: 5px;"><b>👤 Name:</b> <span style="color: #FF5A5F; font-weight: 600;">Dilip</span></div>
            <div style="font-size: 0.95rem; margin-bottom: 5px;"><b>🆔 UID:</b> <span style="font-family: monospace; font-weight: 600;">26BCS12620</span></div>
            <div style="font-size: 0.95rem;"><b>🏫 Section:</b> 508 A</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Project Brief Overview
    st.markdown("### 📋 Project Overview")
    st.markdown(
        """
        An end-to-end Machine Learning regression solution that predicts food delivery duration (in minutes) by modeling:
        - **Geodesic Distance** via the Haversine formula
        - **Live Traffic Congestion** (Low, Medium, High, Jam)
        - **Environmental & Operational Factors** (Weather, Vehicle type, Delivery partner ratings)
        
        Trained on real-world Kaggle delivery records using an optimized **Random Forest Regressor**.
        """
    )

    # Load metrics if available
    metrics_file = "models/metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("R² Score", f"{metrics.get('R2_Score', 0.81)*100:.1f}%")
        col_m2.metric("MAE", f"{metrics.get('MAE', 3.26)} min")
        st.metric("RMSE", f"{metrics.get('RMSE', 4.12)} min")

    st.markdown("---")
    st.markdown("### 💡 Quick Presets")
    if st.button("🍕 Nearby Snack (Low Traffic)"):
        st.session_state["distance"] = 2.5
        st.session_state["traffic"] = "Low"
        st.session_state["city"] = "Urban"
        st.session_state["order_type"] = "Snack"
    if st.button("🍛 Lunch Hour (Medium Traffic)"):
        st.session_state["distance"] = 6.0
        st.session_state["traffic"] = "Medium"
        st.session_state["city"] = "Metropolitian"
        st.session_state["order_type"] = "Meal"
    if st.button("🌧️ Rainy Peak Rush (Jam)"):
        st.session_state["distance"] = 12.0
        st.session_state["traffic"] = "Jam"
        st.session_state["city"] = "Metropolitian"
        st.session_state["weather"] = "Stormy"

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-header">🛵 Food Delivery Time Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict accurate delivery times using Random Forest regression based on distance, traffic, restaurant location, and order attributes.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🚀 Live Prediction", "📊 Model Visualizations", "ℹ️ Methodology"])

with tab1:
    st.markdown("#### Enter Delivery Parameters")

    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.slider(
            "📍 Delivery Distance (in kilometers):",
            min_value=0.5,
            max_value=25.0,
            value=st.session_state.get("distance", 5.0),
            step=0.1,
            help="Calculated using Haversine formula from restaurant coordinates to customer location.",
        )

        traffic = st.selectbox(
            "🚦 Road Traffic Density:",
            options=["Low", "Medium", "High", "Jam"],
            index=["Low", "Medium", "High", "Jam"].index(st.session_state.get("traffic", "Medium")),
            help="Real-time road traffic intensity level.",
        )

        city = st.selectbox(
            "🏙️ City Type / Area:",
            options=["Metropolitian", "Urban", "Semi-Urban"],
            index=["Metropolitian", "Urban", "Semi-Urban"].index(st.session_state.get("city", "Metropolitian")),
        )

        order_type = st.selectbox(
            "🍽️ Type of Order:",
            options=["Meal", "Snack", "Drinks", "Buffet"],
            index=["Meal", "Snack", "Drinks", "Buffet"].index(st.session_state.get("order_type", "Meal")),
        )

    with col2:
        weather = st.selectbox(
            "🌤️ Weather Conditions:",
            options=["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"],
            index=["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"].index(st.session_state.get("weather", "Sunny")),
        )

        vehicle_type = st.selectbox(
            "🛵 Delivery Vehicle Type:",
            options=["motorcycle", "scooter", "electric_scooter", "bicycle"],
            index=0,
        )

        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            rating = st.slider("⭐ Partner Rating:", 3.0, 5.0, 4.7, 0.1)
        with col_sub2:
            age = st.slider("👤 Partner Age:", 18, 50, 28, 1)

        festival = st.selectbox("🎉 Festival / Holiday Season?", ["No", "Yes"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⏱️ Calculate Estimated Delivery Time", type="primary", use_container_width=True)

    if predict_clicked:
        predicted_time = predictor.predict_custom(
            distance_km=distance_km,
            traffic=traffic,
            city=city,
            order_type=order_type,
            vehicle_type=vehicle_type,
            delivery_person_age=age,
            delivery_person_ratings=rating,
            festival=festival,
            weather=weather,
        )

        # ETA Badge & description
        if predicted_time < 20:
            badge = "🟢 Express Delivery"
            color = "#10B981"
            note = "Optimal conditions! Short distance and light traffic allow rapid dispatch."
        elif predicted_time <= 35:
            badge = "🟡 Standard Delivery"
            color = "#F59E0B"
            note = "Standard delivery window. Moderate distance or normal city traffic."
        else:
            badge = "🔴 Delayed Delivery Window"
            color = "#EF4444"
            note = "Adverse conditions detected (heavy traffic, severe weather, or extended distance)."

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size: 0.95rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 1px;">
                    {badge}
                </div>
                <div class="result-number">
                    {predicted_time:.0f} <span style="font-size: 1.5rem; font-weight: 500; color: #555;">minutes</span>
                </div>
                <p style="color: #444; font-size: 1rem; margin-top: 0.5rem;">{note}</p>
                <div style="margin-top: 1rem;">
                    <span class="metric-chip">📍 Distance: {distance_km} km</span>
                    <span class="metric-chip">🚦 Traffic: {traffic}</span>
                    <span class="metric-chip">🏙️ {city}</span>
                    <span class="metric-chip">🌤️ {weather}</span>
                    <span class="metric-chip">🍽️ {order_type}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab2:
    st.markdown("#### 📈 Diagnostic Model Visualizations")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        if os.path.exists("plots/feature_importance.png"):
            st.image("plots/feature_importance.png", caption="Top 10 Feature Importances (Random Forest)", use_container_width=True)
    with col_p2:
        if os.path.exists("plots/actual_vs_predicted.png"):
            st.image("plots/actual_vs_predicted.png", caption="Actual vs. Predicted Delivery Time (R² = 0.81)", use_container_width=True)

    if os.path.exists("plots/residual_distribution.png"):
        st.image("plots/residual_distribution.png", caption="Residual Error Distribution (MAE = 3.26 min)", use_container_width=True)

with tab3:
    st.markdown("#### 🔬 Methodological Architecture")
    st.markdown(
        """
        1. **Feature Engineering**:
           - **Haversine Distance**: Computes great-circle distance between restaurant and destination GPS coordinates:
             $$\\Delta \\sigma = 2 \\arcsin \\sqrt{\\sin^2(\\Delta \\text{lat}/2) + \\cos(\\text{lat}_1)\\cos(\\text{lat}_2)\\sin^2(\\Delta \\text{lon}/2)}$$
           - **Traffic Ordinality**: Maps traffic levels sequentially (`Low: 1`, `Medium: 2`, `High: 3`, `Jam: 4`).
        2. **Machine Learning Model**:
           - **Random Forest Regressor** ensemble with 100 decision trees.
           - Mitigates variance by bagging bootstrapped samples and random feature subsets.
        3. **Validation**:
           - 80% Train (36,474 samples) / 20% Test (9,119 samples) split.
           - MAE = 3.26 min, RMSE = 4.12 min, $R^2 = 80.7\%$.
        """
    )

