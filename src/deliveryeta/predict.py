"""Load the final tuned model and predict delivery time from user-facing inputs.

Used by the Streamlit app. Handles the derived-feature computation
(Traffic_Rank, Distance_x_Traffic, Is_Rush_Hour) internally, so the
caller only needs to supply the "natural" fields a person would enter
in a form.
"""

from functools import lru_cache

import joblib
import pandas as pd

from deliveryeta.paths import MODEL_DIR

MODEL_PATH = MODEL_DIR / "xgboost_tuned_pipeline.joblib"

TRAFFIC_RANK = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
RUSH_HOURS = {12, 13, 19, 20, 21}

VALID_WEATHER = {"Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"}
VALID_TRAFFIC = {"Low", "Medium", "High", "Jam"}
VALID_VEHICLE = {"motorcycle", "scooter", "electric_scooter", "bicycle"}
VALID_ORDER_TYPE = {"Snack", "Meal", "Drinks", "Buffet"}
VALID_FESTIVAL = {"Yes", "No"}


class PredictionInputError(ValueError):
    """Raised when the input payload is missing a field or has an invalid value."""


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 04_hyperparameter_tuning.ipynb first."
        )
    return joblib.load(MODEL_PATH)


def _validate(payload: dict) -> None:
    required = [
        "Distance_km", "Weatherconditions", "Road_traffic_density",
        "Type_of_order", "Type_of_vehicle", "Festival", "City",
        "Delivery_person_Age", "Delivery_person_Ratings",
        "Vehicle_condition", "multiple_deliveries", "Order_Hour",
    ]
    missing = [f for f in required if f not in payload or payload[f] in (None, "")]
    if missing:
        raise PredictionInputError(f"Missing required field(s): {', '.join(missing)}")

    if payload["Weatherconditions"] not in VALID_WEATHER:
        raise PredictionInputError(f"Weatherconditions must be one of {VALID_WEATHER}")
    if payload["Road_traffic_density"] not in VALID_TRAFFIC:
        raise PredictionInputError(f"Road_traffic_density must be one of {VALID_TRAFFIC}")
    if payload["Type_of_vehicle"] not in VALID_VEHICLE:
        raise PredictionInputError(f"Type_of_vehicle must be one of {VALID_VEHICLE}")
    if payload["Type_of_order"] not in VALID_ORDER_TYPE:
        raise PredictionInputError(f"Type_of_order must be one of {VALID_ORDER_TYPE}")
    if payload["Festival"] not in VALID_FESTIVAL:
        raise PredictionInputError(f"Festival must be one of {VALID_FESTIVAL}")

    if not (0 < payload["Distance_km"] <= 50):
        raise PredictionInputError("Distance_km must be between 0 and 50 km")
    if not (1 <= payload["Delivery_person_Ratings"] <= 5):
        raise PredictionInputError("Delivery_person_Ratings must be between 1 and 5")
    if not (15 <= payload["Delivery_person_Age"] <= 65):
        raise PredictionInputError("Delivery_person_Age must be between 15 and 65")
    if payload["multiple_deliveries"] not in (0, 1, 2, 3):
        raise PredictionInputError("multiple_deliveries must be 0, 1, 2 or 3")
    if payload["Vehicle_condition"] not in (0, 1, 2, 3):
        raise PredictionInputError("Vehicle_condition must be 0, 1, 2 or 3")
    if not (0 <= payload["Order_Hour"] <= 23):
        raise PredictionInputError("Order_Hour must be between 0 and 23")


def _build_row(payload: dict) -> pd.DataFrame:
    traffic_rank = TRAFFIC_RANK[payload["Road_traffic_density"]]
    is_rush_hour = int(payload["Order_Hour"] in RUSH_HOURS)
    is_weekend = int(payload.get("Is_Weekend", 0))

    row = {
        "Weatherconditions": payload["Weatherconditions"],
        "Road_traffic_density": payload["Road_traffic_density"],
        "Type_of_order": payload["Type_of_order"],
        "Type_of_vehicle": payload["Type_of_vehicle"],
        "Festival": payload["Festival"],
        "City": payload.get("City", "Metropolitian"),
        "Delivery_person_Age": payload["Delivery_person_Age"],
        "Delivery_person_Ratings": payload["Delivery_person_Ratings"],
        "Vehicle_condition": payload["Vehicle_condition"],
        "multiple_deliveries": payload["multiple_deliveries"],
        "Distance_km": payload["Distance_km"],
        "Order_Hour": payload["Order_Hour"],
        "Is_Weekend": is_weekend,
        "Is_Rush_Hour": is_rush_hour,
        "Traffic_Rank": traffic_rank,
        "Distance_x_Traffic": payload["Distance_km"] * traffic_rank,
    }
    return pd.DataFrame([row])


def predict(payload: dict) -> float:
    """Predict delivery time in minutes from a dict of user-facing inputs.

    Required keys: Distance_km, Weatherconditions, Road_traffic_density,
    Type_of_order, Type_of_vehicle, Festival, City, Delivery_person_Age,
    Delivery_person_Ratings, Vehicle_condition, multiple_deliveries,
    Order_Hour. Optional: Is_Weekend (default 0).

    Raises PredictionInputError on invalid input.
    """
    _validate(payload)
    model = _load_model()
    row = _build_row(payload)
    prediction = model.predict(row)[0]
    return round(float(prediction), 1)
