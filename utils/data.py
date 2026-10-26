"""
Loading data/models once (cached) and turning a form entry into a prediction.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (CLEAN_PARQUET, CLEAN_CSV, MODELS_DIR, REPORTS_DIR,  # noqa: E402
                    REFERENCE_YEAR, FORECAST_YEARS)
from preprocessing import add_features  # noqa: E402

# approximate city centres, for the map
CITY_COORDS = {
    "Mumbai": (19.076, 72.878), "Pune": (18.520, 73.857), "Nagpur": (21.146, 79.088),
    "Bangalore": (12.972, 77.595), "Mysore": (12.296, 76.639), "New Delhi": (28.614, 77.209),
    "Dwarka": (28.592, 77.046), "Hyderabad": (17.385, 78.487), "Warangal": (17.968, 79.594),
    "Chennai": (13.083, 80.270), "Coimbatore": (11.017, 76.956), "Ahmedabad": (23.023, 72.571),
    "Surat": (21.170, 72.831), "Kolkata": (22.573, 88.364), "Durgapur": (23.520, 87.312),
    "Noida": (28.535, 77.391), "Lucknow": (26.847, 80.947), "Jaipur": (26.912, 75.787),
    "Jodhpur": (26.239, 73.024), "Ludhiana": (30.901, 75.857), "Amritsar": (31.634, 74.872),
    "Kochi": (9.931, 76.267), "Trivandrum": (8.524, 76.937), "Indore": (22.720, 75.858),
    "Bhopal": (23.260, 77.413), "Vishakhapatnam": (17.687, 83.218), "Vijayawada": (16.506, 80.648),
    "Patna": (25.594, 85.138), "Gaya": (24.796, 85.008), "Bhubaneswar": (20.296, 85.825),
    "Cuttack": (20.463, 85.883), "Guwahati": (26.144, 91.736), "Silchar": (24.833, 92.779),
    "Ranchi": (23.344, 85.310), "Jamshedpur": (22.805, 86.203), "Raipur": (21.251, 81.630),
    "Bilaspur": (22.080, 82.156), "Dehradun": (30.317, 78.032), "Haridwar": (29.946, 78.164),
    "Gurgaon": (28.460, 77.027), "Faridabad": (28.408, 77.318), "Chandigarh": (30.733, 76.779),
    "Mangalore": (12.914, 74.856),
}


@st.cache_data(show_spinner="Loading listings…")
def load_data() -> pd.DataFrame:
    if CLEAN_PARQUET.exists():
        return pd.read_parquet(CLEAN_PARQUET)
    return pd.read_csv(CLEAN_CSV)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    clf = joblib.load(MODELS_DIR / "classifier.joblib")
    reg = joblib.load(MODELS_DIR / "regressor.joblib")
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    return clf, reg, meta


@st.cache_data
def load_city_reference() -> dict:
    return json.loads((REPORTS_DIR / "city_reference.json").read_text())


@st.cache_data
def load_mlflow_runs() -> pd.DataFrame | None:
    p = REPORTS_DIR / "mlflow_runs.csv"
    return pd.read_csv(p) if p.exists() else None


def models_ready() -> bool:
    return (MODELS_DIR / "classifier.joblib").exists() and (MODELS_DIR / "regressor.joblib").exists()


# ---------------------------------------------------------------------------
def build_row(form: dict) -> pd.DataFrame:
    """Raw form values -> one row with the same engineered columns as training."""
    amen = form.pop("Amenities_list", [])
    row = pd.DataFrame([{
        **form,
        "Amenities": ", ".join(amen),
        "Age_of_Property": REFERENCE_YEAR - int(form["Year_Built"]),
        "Price_per_SqFt": round(form["Price_in_Lakhs"] * 1e5 / max(form["Size_in_SqFt"], 1)),
    }])
    return add_features(row, load_city_reference())


def predict(form: dict) -> dict:
    clf, reg, meta = load_models()
    row = build_row(dict(form))
    X = row[meta["features"]]

    prob = float(clf.predict_proba(X)[0, 1])
    future = float(reg.predict(X)[0])
    price = float(row["Price_in_Lakhs"].iloc[0])

    best = meta["regression"]["best"]
    interval = next(r["interval"] for r in meta["regression"]["results"] if r["model"] == best)
    cagr = (future / price) ** (1 / FORECAST_YEARS) - 1 if price > 0 else 0.0

    return {
        "row": row, "X": X, "prob": prob, "good": prob >= 0.5,
        "price_now": price, "price_5y": future,
        "low_5y": future * interval["low"], "high_5y": future * interval["high"],
        "cagr": cagr,
    }


def explain(X: pd.DataFrame) -> pd.Series | None:
    """
    Per-feature push for this one property (XGBoost's built-in SHAP values,
    in log-odds). Positive = pushes towards 'Good investment'.
    Returns None if the champion model isn't XGBoost.
    """
    clf, _, meta = load_models()
    model = clf.named_steps["model"]
    if not hasattr(model, "get_booster"):
        return None
    import xgboost as xgb

    prep = clf.named_steps["prep"]
    Xt = prep.transform(X)
    contrib = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    names = prep.get_feature_names_out()
    cats = meta["cat_features"]
    base = []
    for n in names:
        kind, col = n.split("__", 1)
        if kind == "cat":
            col = next(c for c in cats if col.startswith(c + "_") or col == c)
        base.append(col)
    return pd.Series(contrib, index=base).groupby(level=0).sum().sort_values(key=np.abs, ascending=False)


PRETTY = {
    "Price_vs_City": "Price vs city average", "Infrastructure_Score": "Infrastructure score",
    "Public_Transport_Accessibility": "Public transport", "Transport_Score": "Public transport (score)",
    "Age_of_Property": "Age of property", "Availability_Status": "Availability",
    "Is_Ready_To_Move": "Ready to move", "Amenity_Count": "Number of amenities",
    "Price_per_SqFt": "Price per sq ft", "Price_in_Lakhs": "Price", "Size_in_SqFt": "Size",
    "Nearby_Schools": "Nearby schools", "Nearby_Hospitals": "Nearby hospitals",
    "School_Density_Score": "School density", "Parking_Space": "Parking", "Size_per_BHK": "Size per BHK",
    "Floor_Ratio": "Floor position", "Floor_No": "Floor", "Total_Floors": "Total floors",
    "Property_Type": "Property type", "Furnished_Status": "Furnishing", "Owner_Type": "Seller type",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("Has_", "Has ").replace("_", " "))
