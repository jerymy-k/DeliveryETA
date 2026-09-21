from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

DELIVERY_DATA_B = BRONZE_DIR / "delivery_data.csv"
DELIVERY_DATA_S = SILVER_DIR / "delivery_data.csv"
DELIVERY_DATA_G = GOLD_DIR / "delivery_data.csv"