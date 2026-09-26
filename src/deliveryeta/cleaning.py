import numpy as np
import pandas as pd


GPS_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

NUMERIC_IMPUTE_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
]

CATEGORICAL_IMPUTE_COLUMNS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Festival",
    "City",
]

COLUMNS_TO_DROP = ["ID", "Delivery_person_ID"]


def _clean_strings_and_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, remove the 'conditions ' prefix, and convert the
    literal text 'NaN' into real pandas NaN."""
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        "conditions ", "", regex=False
    )
    df = df.replace("NaN", np.nan)
    return df


def _clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the numeric value from 'Time_taken(min)' (e.g. '(min) 24' -> 24)."""
    df["Time_taken(min)"] = df["Time_taken(min)"].str.extract(r"(\d+)").astype(float)
    return df


def _repair_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Fix sign errors on restaurant coordinates, drop unrepairable
    placeholder (~0,~0,~0,~0) rows."""
    placeholder_mask = (df[GPS_COLUMNS].abs() < 0.5).all(axis=1)

    df["Restaurant_latitude"] = df["Restaurant_latitude"].abs()
    df["Restaurant_longitude"] = df["Restaurant_longitude"].abs()

    df = df.loc[~placeholder_mask].reset_index(drop=True)
    return df


def _fix_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Invalidate ratings above 5 (impossible on a 1-5 scale); they are
    imputed later along with genuine missing values."""
    ratings_numeric = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    df["Delivery_person_Ratings"] = ratings_numeric.where(ratings_numeric <= 5, np.nan)
    return df


def _cast_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_IMPUTE_COLUMNS:
        df[column] = df[column].fillna(df[column].median())
    for column in CATEGORICAL_IMPUTE_COLUMNS:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def _fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Order_Date to datetime, and reconstruct missing Time_Orderd
    values from Time_Order_picked minus the median prep-time gap."""
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")

    order_dt = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
    picked_dt = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S", errors="coerce")
    prep_minutes = ((picked_dt - order_dt).dt.total_seconds() / 60).mod(1440)
    median_prep = prep_minutes.median()

    missing_mask = df["Time_Orderd"].isna()
    estimated = picked_dt[missing_mask] - pd.to_timedelta(median_prep, unit="m")
    df.loc[missing_mask, "Time_Orderd"] = estimated.dt.strftime("%H:%M:%S")
    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw delivery dataframe.

    Parameters
    ----------
    df_raw : pd.DataFrame
        The raw dataframe as loaded from delivery_data.csv (20 columns).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe (18 columns, 0 missing values).
    """
    df = df_raw.copy()
    df = _clean_strings_and_missing_markers(df)
    df = _clean_target(df)
    df = _repair_gps(df)
    df = _fix_ratings(df)
    df = _cast_numeric_types(df)
    df = _impute_missing(df)
    df = _fix_timestamps(df)
    df = df.drop_duplicates()
    df = df.drop(columns=COLUMNS_TO_DROP)
    return df
