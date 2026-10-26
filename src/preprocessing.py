"""
Step 1 - cleaning + feature engineering.

    python src/preprocessing.py

Reads data/raw/india_housing_prices.csv(.gz) and writes a cleaned, model-ready
copy to data/processed/ (gzipped CSV for the deliverable, parquet for the app).

The feature functions at the bottom are also imported by the Streamlit app,
so a property typed into the form goes through exactly the same steps as
the training data did.
"""

import json

import numpy as np
import pandas as pd

from config import (
    RAW_CSV, CLEAN_CSV, CLEAN_PARQUET, REPORTS_DIR, REFERENCE_YEAR, FORECAST_YEARS,
    AMENITY_LIST, TRANSPORT_LEVELS, BASE_GROWTH, MIN_GROWTH, MAX_GROWTH,
    GOOD_GROWTH_THRESHOLD, MAX_PRICE_VS_CITY, RANDOM_STATE,
)

CATEGORICAL = ["State", "City", "Locality", "Property_Type", "Furnished_Status",
               "Public_Transport_Accessibility", "Facing", "Owner_Type",
               "Availability_Status"]
NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
           "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _to_flag(series: pd.Series) -> pd.Series:
    """Yes/No, True/False, 1/0 or a count of spots -> clean 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes", "y", "true", "1", "available"]).astype(int)


def _tidy_text(series: pd.Series) -> pd.Series:
    # "semi furnished", "Semi-Furnished " etc. all collapse to one spelling
    return (series.astype("string").str.strip()
            .str.replace(r"\s+", " ", regex=True))


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------
def clean(raw: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = raw.copy()
    log["rows_raw"] = int(len(df))

    # duplicates - first exact copies, then the same listing under a new ID
    before = len(df)
    df = df.drop_duplicates()
    cols_wo_id = [c for c in df.columns if c != "ID"]
    df = df.drop_duplicates(subset=cols_wo_id)
    log["duplicates_removed"] = int(before - len(df))

    log["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}

    for col in CATEGORICAL + ["Amenities"]:
        if col in df:
            df[col] = _tidy_text(df[col])
    if "Furnished_Status" in df:
        df["Furnished_Status"] = df["Furnished_Status"].str.replace(
            r"(?i)^semi.?furnished$", "Semi-furnished", regex=True)

    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # numeric gaps: the city median is a fairer guess than the national one
    for col in NUMERIC:
        if col in df and df[col].isna().any():
            df[col] = df[col].fillna(df.groupby("City")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())

    # categorical gaps: most common value in that city
    for col in CATEGORICAL:
        if col in df and df[col].isna().any():
            city_mode = df.groupby("City")[col].transform(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
            df[col] = df[col].fillna(city_mode).fillna(df[col].mode().iloc[0])

    # an empty amenities cell means "none listed", not "unknown"
    df["Amenities"] = df["Amenities"].fillna("")

    for col in ["Parking_Space", "Security"]:
        df[col] = _to_flag(df[col])

    # sanity fixes
    df["Total_Floors"] = df["Total_Floors"].clip(lower=1)
    bad_floor = df["Floor_No"] > df["Total_Floors"]
    log["floor_above_building_fixed"] = int(bad_floor.sum())
    df.loc[bad_floor, "Floor_No"] = df.loc[bad_floor, "Total_Floors"]

    df["Year_Built"] = df["Year_Built"].clip(upper=REFERENCE_YEAR)
    df["Age_of_Property"] = REFERENCE_YEAR - df["Year_Built"]

    int_cols = ["BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
                "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]
    df[int_cols] = df[int_cols].round().astype(int)

    # The provided Price_per_SqFt is in lakhs/sqft (0.07 etc.), which is
    # hard to read and sometimes doesn't match price/size. I recompute it in
    # plain rupees so it's consistent and easy to talk about.
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(0)

    # outliers - flag with the IQR rule (kept for EDA), cap for modelling
    for col in ["Price_per_SqFt", "Size_in_SqFt"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = (df[col] < lo) | (df[col] > hi)
        df[f"{col}_Outlier"] = flag.astype(int)
        log[f"{col}_outliers"] = int(flag.sum())

    log["missing_after"] = int(df[CATEGORICAL + NUMERIC].isna().sum().sum())
    return df


# ---------------------------------------------------------------------------
# features (shared with the app)
# ---------------------------------------------------------------------------
def city_reference(df: pd.DataFrame) -> dict:
    """City medians the app needs to judge a single new property."""
    return {
        "city_median_ppsf": df.groupby("City")["Price_per_SqFt"].median().round(0).to_dict(),
        "national_median_ppsf": float(df["Price_per_SqFt"].median()),
    }


def add_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    out = df.copy()

    amen = out["Amenities"].fillna("").astype(str).str.lower()
    for a in AMENITY_LIST:
        out[f"Has_{a}"] = amen.str.contains(a.lower(), regex=False).astype(int)
    out["Amenity_Count"] = out[[f"Has_{a}" for a in AMENITY_LIST]].sum(axis=1)

    out["Transport_Score"] = (out["Public_Transport_Accessibility"]
                              .map(TRANSPORT_LEVELS).fillna(2).astype(int))

    # 0-10 scores. Schools/hospitals in the source run 1-10, so this is mostly
    # a rescale, but it keeps the app honest if someone types 0.
    out["School_Density_Score"] = (out["Nearby_Schools"].clip(0, 10)).round(1)
    out["Hospital_Density_Score"] = (out["Nearby_Hospitals"].clip(0, 10)).round(1)
    out["Infrastructure_Score"] = (10 * (
        0.35 * out["Nearby_Schools"].clip(0, 10) / 10
        + 0.35 * out["Nearby_Hospitals"].clip(0, 10) / 10
        + 0.30 * (out["Transport_Score"] - 1) / 2)).round(2)

    out["Floor_Ratio"] = (out["Floor_No"] / out["Total_Floors"].clip(lower=1)).round(3)
    out["Size_per_BHK"] = (out["Size_in_SqFt"] / out["BHK"].clip(lower=1)).round(0)
    out["Is_Ready_To_Move"] = out["Availability_Status"].astype(str).str.lower() \
        .str.startswith("ready").astype(int)

    city_med = out["City"].map(ref["city_median_ppsf"]).fillna(ref["national_median_ppsf"])
    out["City_Median_PPSF"] = city_med
    out["Price_vs_City"] = (out["Price_per_SqFt"] / city_med).round(3)
    return out


def expected_growth(df: pd.DataFrame, add_noise: bool = False) -> pd.Series:
    """
    Expected appreciation per year. Each term is small on purpose - nobody
    should believe a model that says a flat goes up 25% a year.
    """
    value_gap = (1 - df["Price_vs_City"]).clip(-0.5, 0.5)       # cheaper than city = room to catch up
    transport_bonus = df["Transport_Score"].map({1: -0.004, 2: 0.004, 3: 0.010}).fillna(0)
    liquid_size = df["BHK"].isin([2, 3]).astype(int)            # 2-3 BHK resell fastest

    g = (BASE_GROWTH
         + 0.005 * (df["Infrastructure_Score"] - 5)
         + 0.030 * value_gap
         + transport_bonus
         + 0.010 * (1 - df["Is_Ready_To_Move"])                 # new launches appreciate faster
         - 0.0006 * df["Age_of_Property"].clip(0, 40)
         + 0.0025 * df["Amenity_Count"]
         + 0.003 * df["Security"] + 0.002 * df["Parking_Space"]
         + 0.003 * liquid_size)

    if add_noise:
        # real markets don't follow a formula exactly; a little noise keeps
        # the models from just memorising the rule
        rng = np.random.default_rng(RANDOM_STATE)
        g = g + rng.normal(0, 0.006, len(g))

    return g.clip(MIN_GROWTH, MAX_GROWTH).round(4)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Expected_Annual_Growth"] = expected_growth(out, add_noise=True)
    out["Future_Price_5Y"] = (out["Price_in_Lakhs"]
                              * (1 + out["Expected_Annual_Growth"]) ** FORECAST_YEARS).round(2)
    out["Good_Investment"] = ((out["Expected_Annual_Growth"] >= GOOD_GROWTH_THRESHOLD)
                              & (out["Price_vs_City"] <= MAX_PRICE_VS_CITY)).astype(int)
    return out


# ---------------------------------------------------------------------------
def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"Can't find the raw data in {RAW_CSV.parent}. Download "
                         "india_housing_prices.csv from the course link and put it there.")

    log = {}
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns")

    df = clean(raw, log)
    ref = city_reference(df)
    df = add_features(df, ref)
    df = add_targets(df)

    log["rows_clean"] = int(len(df))
    log["good_investment_rate"] = round(float(df["Good_Investment"].mean()), 4)
    log["avg_expected_growth"] = round(float(df["Expected_Annual_Growth"].mean()), 4)

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    df.to_parquet(CLEAN_PARQUET, index=False)
    (REPORTS_DIR / "city_reference.json").write_text(json.dumps(ref, indent=2))
    (REPORTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))

    print(f"Removed {log['duplicates_removed']:,} duplicates, "
          f"fixed {log['floor_above_building_fixed']} impossible floor numbers")
    print(f"Clean rows: {len(df):,} | Good Investment share: {log['good_investment_rate']:.1%} "
          f"| avg growth: {log['avg_expected_growth']:.2%}/yr")
    print(f"Saved -> {CLEAN_CSV.relative_to(CLEAN_CSV.parents[2])}")


if __name__ == "__main__":
    main()
