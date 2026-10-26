# Real Estate Investment Advisor: project report

**Predicting property profitability and future value** · Shyam Jagani

---

## 1. The problem

Someone looking at an Indian property listing usually wants two answers:

1. **Is this a good investment?** A yes/no call, with a sense of how confident it is.
2. **What will it be worth in five years?** A number, with a sensible range around it.

The brief asks for a classifier for (1), a regressor for (2), experiment tracking in MLflow, and a Streamlit app where an investor can enter a property, filter listings and see visual insights.

**Data:** `india_housing_prices.csv`, with 250,000 listings × 23 columns across 20 states, 42 cities and 500 localities. It covers price, size, BHK, age, floors, nearby schools and hospitals, transport access, parking, security, amenities, facing, seller type and availability.

---

## 2. Business use cases

| Who | How they'd use it |
|---|---|
| **Individual investors** | Check one property before committing: verdict, 5-year value, and what's driving both |
| **Home buyers** | Filter by budget, size and BHK, then sort by expected growth to find high-return homes in developing areas |
| **Real estate platforms** | Score every listing automatically and show a "good investment" badge with the reasons behind it |
| **Trust and transparency** | Each verdict comes with its confidence, a likely price range and the features behind the call, so it's data-backed rather than a black box |

---

## 3. Methodology

### 3.1 Data cleaning (`src/preprocessing.py`)

| Check | What I found | What I did |
|---|---|---|
| Duplicates | 0 exact, 0 re-posted under a new ID | Pipeline drops both kinds |
| Missing values | None in the course file | City-level median (numbers) or mode (categories) imputation built in anyway |
| Floor numbers | **116,304 rows (46.5%)** list a floor above the building's total floors | Capped at the top floor |
| `Price_per_SqFt` | Stored in lakhs per sq ft (e.g. 0.09) and not always equal to price ÷ size | Recalculated in rupees: price × 100,000 ÷ size |
| Outliers | 19,724 listings (7.9%) outside 1.5 × IQR on price per sq ft; none on size | Flagged, not dropped. They can be hidden in the app |
| Text values | Consistent spelling | Whitespace/case normalised; Yes/No mapped to 1/0 |

Encoding (one-hot for categories) and scaling (standard scaler for numbers) live **inside the model pipeline**, so a property typed into the app goes through exactly the same transformation as the training data.

### 3.2 Feature engineering

| Feature | Definition |
|---|---|
| `Price_per_SqFt` | ₹ per sq ft (recalculated) |
| `Price_vs_City` | Listing's ₹/sq ft ÷ its city's median. Below 1 means cheaper than the local market |
| `School_Density_Score` | Nearby schools on a 0-10 scale |
| `Infrastructure_Score` | 10 × (0.35 schools + 0.35 hospitals + 0.30 transport), each scaled 0-1 |
| `Transport_Score` | Low / Medium / High → 1 / 2 / 3 |
| `Amenity_Count`, `Has_Gym` … `Has_Clubhouse` | Parsed from the comma-separated amenities text |
| `Floor_Ratio`, `Size_per_BHK` | Floor position in the building; space per bedroom |
| `Is_Ready_To_Move` | 1 if available now |

### 3.3 Target variables

The data has current prices only, with no price history, so both targets are derived from an **expected annual appreciation rate** built from domain rules:

```
rate = 6% base
     + 0.5% × (Infrastructure_Score − 5)
     + 3%   × (1 − Price_vs_City)        # value gap, capped at ±50%
     + transport: −0.4% low / +0.4% medium / +1.0% high
     + 1.0% if under construction        # new-launch premium
     − 0.06% per year of age
     + 0.25% per amenity, +0.3% security, +0.2% parking, +0.3% for 2-3 BHK
     + small random noise (σ = 0.6%)
     clipped to 2% … 14%
```

- **`Future_Price_5Y`** = `Price_in_Lakhs × (1 + rate)^5`. This is the regression target.
- **`Good_Investment`** = `rate ≥ 7.5%` **and** `Price_vs_City ≤ 1.15`. This is the classification target. **34.4%** of listings qualify.

### 3.4 Modelling (`src/train.py`)

- 80 / 20 train-test split, stratified on `Good_Investment`. 32 input features (Locality is excluded: 500 levels add noise, and City carries the location signal).
- **Classification:** Logistic Regression, Random Forest, XGBoost. Best chosen by **F1**, because the classes are imbalanced.
- **Regression:** Linear Regression, Random Forest, XGBoost. Best chosen by **RMSE**.
- **MLflow** (`sqlite:///mlflow.db`, experiment `real-estate-investment-advisor`) logs every run's parameters, metrics, confusion matrix or actual-vs-predicted plot, feature-importance CSV and the fitted pipeline with its signature. The winners are registered as `RealEstate_GoodInvestment_Classifier` and `RealEstate_FuturePrice_Regressor` with the alias `champion`.

## 4. Results

Test set: 50,000 held-out listings (20%).

### 4.1 Classification: Good Investment

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Train time |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9145 | 0.8770 | 0.8745 | 0.8758 | 0.9754 | 3 s |
| Random Forest | 0.9342 | 0.9008 | 0.9090 | 0.9049 | 0.9851 | 18 s |
| **XGBoost (champion)** | **0.9440** | **0.9175** | **0.9202** | **0.9188** | **0.9887** | 17 s |

Confusion matrix (XGBoost): 31,351 true "think twice", 15,849 true "good"; 1,426 false alarms and 1,374 missed good buys.

The top drivers are **Price vs city** (23%), **public transport** (18% plus 6% for its numeric score), **ready to move** (10%), **city** (10%, through its median price) and **infrastructure score** (7%). That's the rule set coming back out of the model, which is the sanity check I wanted.

### 4.2 Regression: price in 5 years

| Model | RMSE (₹ L) | MAE (₹ L) | MAPE | R² | Train time |
|---|---|---|---|---|---|
| Linear Regression | 20.74 | 15.98 | 9.99% | 0.9879 | 4 s |
| Random Forest | 13.18 | 9.36 | 2.73% | 0.9951 | 121 s |
| **XGBoost (champion)** | **11.63** | **8.21** | **2.44%** | **0.9962** | 25 s |

Today's price carries most of the weight (61%), as it should. The model's job is the growth on top, which comes from price per sq ft, transport and price vs city. On a median ₹2.54 Cr home, an RMSE of ₹11.6 L is an error of about 4.6% of today's price over a 5-year horizon. 90% of test predictions fall within **−5% / +5%** of the true value, and the app shows that as the likely range.

**Why the scores are this high:** the targets are built from the features with a little noise, so a good model should nearly recover them. Linear regression lags because growth compounds multiplicatively (price × (1 + rate)⁵), which tree models capture and a straight line can't.


---

## 5. EDA findings

The full set, one takeaway per question, is in `reports/eda_findings.md`, and every chart is in `reports/figures/`. The headline is unusual:

**Almost nothing in the raw data moves the price.**

- **Size vs price: r = 0.00.** A 600 sq ft flat is as likely to be listed at ₹4 Cr as a 4,800 sq ft villa. Prices are spread flat between ₹10 L and ₹5 Cr (median ₹2.54 Cr).
- **Location barely matters.** The most and least expensive states (Karnataka vs Punjab) are only 2.5% apart on price per sq ft; the most and least expensive cities are 3.1% apart.
- **Schools, hospitals, furnishing, facing, parking and amenities** each shift price per sq ft by less than 1%.
- **Property type:** independent houses vs villas differ by about 1%.
- **The categories are evenly split:** owner/builder/broker about 33% each; ready-to-move vs under construction 50/50; each BHK about 20%.
- **No real price trend by build year**, either for the top 5 localities or the top cities.

**What this means.** Real housing data never looks like this. Size alone usually explains most of the price. The listings appear to be **randomly generated**, so I report "no effect" plainly rather than reading patterns into noise. It's also why the brief's domain-rule approach to the targets is the right one: the investment signal has to come from engineered rules, not from patterns hidden in the raw prices.

**Where the engineered view does show structure (Step 2 of the brief):**

| Analysis | Finding |
|---|---|
| Infrastructure vs resale value | Best-served areas (score 8-10) are expected to gain **56%** in five years vs **24%** in the weakest |
| Public transport vs investment potential | **52%** of high-access homes are good investments vs **14%** of low-access ones |
| Area vs investment return | r = 0.33. Larger homes do better because price doesn't rise with size, so they're usually priced below their city's median per sq ft |
| Crime-risk proxy (Security) | 36% of secured homes are good investments vs 33% without. There's no crime column, so Security stands in |

---

## 6. The Streamlit app

| Page | Purpose |
|---|---|
| Overview | Animated dusk skyline (towers rise, windows light up, a crane works on a half-built tower), key numbers, cities with the most good buys |
| Investment advisor | Property form, two one-click examples, an animated verdict (house builds itself, verdict stamp, price count-up, growth line with likely range, confidence ring), per-property explanation (XGBoost SHAP values), city comparison, similar listings |
| Find properties | Filters for city, BHK, type, budget, size, good-only, ready-to-move and outliers; sortable results; CSV download |
| Market map | City map coloured by price, growth or good-buy share; state × property-type heatmap; the four Step 2 analyses |
| 20 questions (EDA) | Every brief question with chart and takeaway |
| Model lab | Metrics for all six models, ROC curves, confusion matrix, importances, actual vs predicted, MLflow run log |
| How it works | Cleaning, features, target rules and limits, explained in plain English |

All animations are inline SVG/CSS with no external files, and they respect the "reduce motion" setting.

---

## 7. Limitations and next steps

- **Targets are rule-based.** The models learn my rules very well, which is what the high scores show. They don't learn the real market. With actual resale histories (e.g. registration data), the same pipeline can be retrained on true 5-year price changes with no code changes beyond the target.
- **Synthetic-looking data.** Insights about price drivers here should not be carried over to real Indian markets.
- **No crime data, no sale dates.** Security is used as a crime proxy, and year built as the timeline.
- **Next steps:** real transaction data; city-level growth rates from RBI's Residential Property Price Index; hyper-parameter search logged to MLflow; a REST endpoint serving the registered `champion` models.
