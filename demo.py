"""
demo.py
Interactive Food Delivery Time Prediction Demo.
Run directly to test custom delivery scenarios.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.predict import DeliveryTimePredictor


def main():
    print("=" * 60)
    print(" 🍔 Food Delivery Time Prediction - Interactive Demo 🛵 ")
    print("   Student: Dilip | UID: 26BCS12620 | Section: 508 A")
    print("   Model: Random Forest Regressor")
    print("=" * 60)

    try:
        predictor = DeliveryTimePredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nPresets:")
    print("1. Quick Snack Nearby (2.5 km, Low Traffic, Urban)")
    print("2. Lunch Hour Office Delivery (6.0 km, Medium Traffic, Metropolitian)")
    print("3. Peak Dinner Rush (14.0 km, Jam Traffic, Metropolitian)")
    print("4. Custom Input (Enter your own values)")
    print("0. Exit")

    while True:
        choice = input("\nSelect an option [0-4]: ").strip()
        if choice == "0":
            print("Exiting demo. Have a great day!")
            break
        elif choice == "1":
            pred = predictor.predict_custom(distance_km=2.5, traffic="Low", city="Urban", order_type="Snack")
            print(f"\n>> Result: Estimated Delivery Time is {pred} minutes.")
        elif choice == "2":
            pred = predictor.predict_custom(distance_km=6.0, traffic="Medium", city="Metropolitian", order_type="Meal")
            print(f"\n>> Result: Estimated Delivery Time is {pred} minutes.")
        elif choice == "3":
            pred = predictor.predict_custom(distance_km=14.0, traffic="Jam", city="Metropolitian", order_type="Meal")
            print(f"\n>> Result: Estimated Delivery Time is {pred} minutes.")
        elif choice == "4":
            try:
                dist_input = float(input("Enter distance in kilometers (e.g., 5.5): ").strip())
                print("Traffic options: Low, Medium, High, Jam")
                traffic_input = input("Enter traffic density: ").strip().title()
                print("City options: Urban, Metropolitian, Semi-Urban")
                city_input = input("Enter city type: ").strip().title()

                pred = predictor.predict_custom(
                    distance_km=dist_input,
                    traffic=traffic_input if traffic_input else "Medium",
                    city=city_input if city_input else "Urban",
                )
                print(f"\n>> Result: Estimated Delivery Time is {pred} minutes.")
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
        else:
            print("Invalid choice, please select 0, 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

