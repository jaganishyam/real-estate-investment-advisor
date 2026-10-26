"""
Shared settings for the whole project.

Everything that more than one script needs (paths, the reference year, the
rules behind the "Good Investment" label) lives here, so if I change a
threshold it changes everywhere - the training script and the Streamlit app
can never drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# CSVs are stored gzipped (.csv.gz) so every file stays under GitHub's 25 MB
# upload limit. pandas reads and writes .gz directly, no unzipping needed.
# If you drop the original, uncompressed india_housing_prices.csv into
# data/raw/, that file is used instead.
_RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = next((p for p in [_RAW_DIR / "india_housing_prices.csv",
                            _RAW_DIR / "india_housing_prices.csv.gz"] if p.exists()),
               _RAW_DIR / "india_housing_prices.csv.gz")
CLEAN_CSV = ROOT / "data" / "processed" / "housing_clean.csv.gz"
CLEAN_PARQUET = ROOT / "data" / "processed" / "housing_clean.parquet"  # faster for the app

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "real-estate-investment-advisor"

# The dataset doesn't carry a listing date, so "today" for age calculations
# is pinned to the year the data was collected.
REFERENCE_YEAR = 2025
FORECAST_YEARS = 5
RANDOM_STATE = 42

AMENITY_LIST = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
TRANSPORT_LEVELS = {"Low": 1, "Medium": 2, "High": 3}

# ---------------------------------------------------------------------------
# How I define the targets
# ---------------------------------------------------------------------------
# The raw data has no historic sale prices, so there's nothing to "learn" the
# future from directly. Instead I build an expected yearly appreciation rate
# from things that genuinely move Indian property prices (connectivity,
# social infrastructure, how fairly it's priced vs. the rest of the city,
# age, new-launch premium, amenities). The 5-year price and the Good
# Investment flag are both derived from that rate.

BASE_GROWTH = 0.060          # long-run average for Indian residential, ~6%/yr
MIN_GROWTH, MAX_GROWTH = 0.02, 0.14

GOOD_GROWTH_THRESHOLD = 0.075    # needs >= 7.5% a year expected appreciation...
MAX_PRICE_VS_CITY = 1.15         # ...and must not be >15% pricier than the city median
