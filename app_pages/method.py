import json

import streamlit as st

from config import (BASE_GROWTH, GOOD_GROWTH_THRESHOLD, MAX_PRICE_VS_CITY, REPORTS_DIR,
                    FORECAST_YEARS)
from utils import style

style.page_header("How it works",
                  "What happens between the raw CSV and the verdict on the advisor page.")

log_path = REPORTS_DIR / "cleaning_log.json"
log = json.loads(log_path.read_text()) if log_path.exists() else {}

st.markdown("## 1. Cleaning the data")
if log:
    missing = sum(log.get("missing_before", {}).values())
    style.stat_strip([
        (f"{log['rows_raw']:,}", "Rows in the raw file"),
        (f"{log['duplicates_removed']:,}", "Duplicates removed"),
        (f"{missing:,}", "Missing values filled"),
        (f"{log['floor_above_building_fixed']:,}", "Impossible floor numbers fixed"),
        (f"{log['rows_clean']:,}", "Clean rows"),
    ])
floor_share = (f"{log['floor_above_building_fixed'] / log['rows_clean']:.1%}"
               if log.get("rows_clean") else "Some")
st.markdown(f"""
- **Duplicates** - exact copies first, then the same listing posted under a new ID.
- **Missing values** - the course file has none, but the pipeline handles them anyway: numbers get the median *for that city* (a fairer guess than a national figure), categories get the city's most common value, and an empty amenities cell means "none listed".
- **Price per sq ft** is recalculated in rupees from price and size. The original column was in lakhs per sq ft and didn't always agree with price ÷ size.
- **Floor numbers** - {floor_share} of rows list a floor higher than the building's total floors. Those are capped at the top floor.
- **Outliers** are flagged with the 1.5 × IQR rule but kept, because they could be genuine luxury homes or bargains. You can hide them on the *Find properties* page.
- **Encoding and scaling** happen inside the model pipeline: one-hot encoding for categories and standard scaling for numbers. The app runs a new property through exactly the same steps.
""")

st.markdown("## 2. New features")
st.markdown("""
| Feature | What it captures |
|---|---|
| Price per sq ft (₹) | Normalised price, comparable across sizes |
| Price vs city | This home's ₹/sq ft ÷ its city's median. Below 1 means cheaper than the local market |
| School density score | Nearby schools on a 0-10 scale |
| Infrastructure score | 35% schools + 35% hospitals + 30% public transport, on a 0-10 scale |
| Amenity count + one flag per amenity | Gym, pool, garden, playground, clubhouse |
| Floor ratio, size per BHK | Floor position in the building; how roomy each bedroom is |
| Ready to move | 1 if available now, 0 if under construction |
""")

st.markdown("## 3. Defining the two targets")
st.markdown(f"""
The dataset has today's prices but no price history, so there's nothing that directly says how a
home *will* do. I built an **expected yearly appreciation rate** for every listing from the
things that move Indian property prices:

- starts from a long-run base of **{BASE_GROWTH:.0%} a year**
- **better infrastructure** (schools, hospitals, transport) adds growth
- **priced below the city median** adds growth, because there's room to catch up; overpriced homes lose some
- **under construction** gets a new-launch premium
- **older buildings** lose a little each year of age
- **amenities, security, parking** and the easy-to-resell **2-3 BHK** sizes add small amounts
- a little random noise, because real markets never follow a formula exactly

From that rate:

- **Price in {FORECAST_YEARS} years** = today's price × (1 + rate)<sup>{FORECAST_YEARS}</sup>. This is the regression target.
- **Good Investment** = expected growth of at least **{GOOD_GROWTH_THRESHOLD:.1%} a year** *and* a price no more than **{MAX_PRICE_VS_CITY - 1:.0%} above the city median**. This is the classification target.

The rules live in `src/config.py`, so they can be tuned in one place.
""", unsafe_allow_html=True)

st.markdown("## 4. Models and tracking")
good_rate = f"{log['good_investment_rate']:.0%}" if log.get("good_investment_rate") else "a minority"
st.markdown(f"""
- Split 80 / 20, stratified on the Good Investment label.
- **Classification:** Logistic Regression, Random Forest and XGBoost. Picked by F1, because the classes are imbalanced ({good_rate} of listings are good buys).
- **Regression:** Linear Regression, Random Forest and XGBoost. Picked by RMSE.
- **MLflow** logs every run's parameters, metrics, plots, feature importance and model. The winners go into the Model Registry with a `champion` alias.
- Locality is left out of the models. With 500 levels and only a few hundred rows each, it mostly adds noise, and city already carries the location signal.
""")

st.markdown("## 5. Honest limits")
st.markdown("""
- Both targets come from rules I wrote, so the models are learning those rules. High accuracy shows the models capture them well, not that they predict the real market. With real resale histories, the same pipeline could be retrained on actual price changes.
- The dataset has no crime-rate column, so the Security flag (gated/CCTV/guard) stands in as a crime-risk proxy.
- There are no sale dates, so "price trends" use year built as the timeline.
- **The listings look randomly generated.** Asking price has no relationship with size (r ≈ 0), and average prices differ by only about 3% between the cheapest and most expensive city. Real housing data never looks like that. So the EDA honestly reports "no effect" for most single features, and the investment signal comes from the engineered growth rules rather than from patterns hidden in the raw prices.
""")
