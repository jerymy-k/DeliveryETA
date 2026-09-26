"""Feature engineering for the cleaned delivery dataset (Silver -> Gold).

Refactored from notebooks/02_eda.ipynb. Includes Distance_x_Traffic,
the strongest single predictor found in EDA (correlation 0.448 with
the target, vs. 0.32 for Distance_km alone).
"""

import numpy as np
import pandas as pd


TRAFFIC_RANK = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
RUSH_HOURS = [12, 13, 19, 20, 21]

RAW_COLUMNS_TO_DROP = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
]


def haversine(lat1, lon1, lat2, lon2) -> pd.Series:
    """Straight-line distance in km between two GPS points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return (R * 2 * np.arcsin(np.sqrt(a))).round(2)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned delivery dataframe.

    Expects df to still have the raw GPS columns and Order_Date /
    Time_Orderd (as produced by cleaning.clean_data). Drops those raw
    columns once the derived features are computed.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of cleaning.clean_data).

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features, raw GPS/date/time columns removed.
    """
    df = df.copy()

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S")

    df["Distance_km"] = haversine(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )

    df["Order_Hour"] = df["Time_Orderd"].dt.hour
    df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek
    df["Is_Weekend"] = df["Order_DayOfWeek"].isin([5, 6]).astype(int)
    df["Is_Rush_Hour"] = df["Order_Hour"].isin(RUSH_HOURS).astype(int)

    df["Traffic_Rank"] = df["Road_traffic_density"].map(TRAFFIC_RANK)
    df["Distance_x_Traffic"] = df["Distance_km"] * df["Traffic_Rank"]

    df = df.drop(columns=RAW_COLUMNS_TO_DROP)
    return df
