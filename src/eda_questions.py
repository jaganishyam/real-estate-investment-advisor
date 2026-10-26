"""
The 20 EDA questions from the brief, plus the four analyses listed under
"Step 2" of the approach.

Every function takes the cleaned dataframe and returns (figure, takeaway).
The takeaway is worked out from the data each time - nothing is hard-coded -
so the text stays true if the dataset changes.

Used by src/eda.py (exports charts + a findings file) and by the
"20 Questions" page in the Streamlit app.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_theme import SERIES, SEQUENTIAL, DIVERGING, GOOD, CAUTION, MUTED, money, rupees

SAMPLE_N = 6000   # scatter plots get unreadable (and slow) beyond this


def _sample(df, n=SAMPLE_N):
    return df.sample(min(n, len(df)), random_state=7)


def _pct(x):
    return f"{x:.0%}"


# ---------------------------------------------------------------- 1-5 price & size
def q1_price_distribution(df):
    s = df["Price_in_Lakhs"]
    fig = px.histogram(df, x="Price_in_Lakhs", nbins=60, color_discrete_sequence=[SERIES[0]],
                       labels={"Price_in_Lakhs": "Price (₹ lakhs)"})
    fig.add_vline(s.median(), line_dash="dot", line_color=MUTED,
                  annotation_text=f"median {money(s.median())}", annotation_font_color=MUTED)
    fig.update_layout(title="How property prices are spread", yaxis_title="Listings")
    skew = s.skew()
    shape = ("leans right - a long tail of expensive homes" if skew > 0.5
             else "fairly flat, with no single dominant price band" if abs(skew) < 0.2
             else "roughly symmetric")
    note = (f"Half of all listings sit between {money(s.quantile(.25))} and "
            f"{money(s.quantile(.75))}, with a median of {money(s.median())}. "
            f"The distribution is {shape} (skew {skew:.2f}).")
    return fig, note


def q2_size_distribution(df):
    s = df["Size_in_SqFt"]
    fig = px.histogram(df, x="Size_in_SqFt", nbins=60, color_discrete_sequence=[SERIES[0]],
                       labels={"Size_in_SqFt": "Size (sq ft)"})
    fig.add_vline(s.median(), line_dash="dot", line_color=MUTED,
                  annotation_text=f"median {s.median():,.0f} sq ft", annotation_font_color=MUTED)
    fig.update_layout(title="How property sizes are spread", yaxis_title="Listings")
    note = (f"A typical home is about {s.median():,.0f} sq ft; the middle half ranges from "
            f"{s.quantile(.25):,.0f} to {s.quantile(.75):,.0f} sq ft. "
            f"Smallest {s.min():,.0f}, largest {s.max():,.0f}.")
    return fig, note


def q3_ppsf_by_type(df):
    order = df.groupby("Property_Type")["Price_per_SqFt"].median().sort_values().index.tolist()
    fig = px.box(df, x="Property_Type", y="Price_per_SqFt", points=False,
                 category_orders={"Property_Type": order}, color_discrete_sequence=[SERIES[0]],
                 labels={"Price_per_SqFt": "Price per sq ft (₹)", "Property_Type": ""})
    fig.update_layout(title="Price per sq ft by property type")
    fig.update_yaxes(type="log")   # a few ₹50k+/sq ft listings would squash the boxes otherwise
    med = df.groupby("Property_Type")["Price_per_SqFt"].median().sort_values()
    gap = med.iloc[-1] / med.iloc[0] - 1
    gap = med.iloc[-1] / med.iloc[0] - 1
    note = (f"{med.index[-1]} has the highest median rate at {rupees(med.iloc[-1])}/sq ft, "
            + (f"only {gap:.0%} above {med.index[0].lower()} ({rupees(med.iloc[0])}/sq ft). "
               "Property type barely changes the price per sq ft." if gap < 0.03 else
               f"{gap:.0%} above {med.index[0].lower()} ({rupees(med.iloc[0])}/sq ft)."))
    return fig, note


def q4_size_vs_price(df):
    d = _sample(df)
    r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    fig = px.scatter(d, x="Size_in_SqFt", y="Price_in_Lakhs", opacity=0.35,
                     color_discrete_sequence=[SERIES[0]],
                     labels={"Size_in_SqFt": "Size (sq ft)", "Price_in_Lakhs": "Price (₹ lakhs)"})
    m, c = np.polyfit(df["Size_in_SqFt"], df["Price_in_Lakhs"], 1)
    xs = np.linspace(df["Size_in_SqFt"].min(), df["Size_in_SqFt"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=m * xs + c, mode="lines", name="trend",
                             line=dict(color=SERIES[1], width=2)))
    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))
    fig.update_layout(title="Size vs price", showlegend=False)
    if abs(r) < 0.05:
        note = (f"Surprisingly, no link at all (r = {r:.2f}): a 600 sq ft flat is as likely to "
                f"be listed at ₹4 Cr as a 4,800 sq ft villa. In real markets size is the biggest "
                f"driver of price, so this is a strong hint the listings were generated at "
                f"random. That's why per-sq-ft prices range so widely.")
    else:
        strength = "strong" if abs(r) > .6 else "moderate" if abs(r) > .3 else "weak"
        direction = "positive" if r > 0 else "negative"
        note = (f"There's a {strength} {direction} link (r = {r:.2f}). On average every extra "
                f"100 sq ft adds about {money(m * 100)} to the asking price.")
    return fig, note


def q5_outliers(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Price per sq ft (₹)", "Size (sq ft)"))
    fig.add_trace(go.Box(y=df["Price_per_SqFt"], name="", marker_color=SERIES[0],
                         boxpoints="outliers", marker=dict(size=3, opacity=.3)), 1, 1)
    fig.add_trace(go.Box(y=df["Size_in_SqFt"], name="", marker_color=SERIES[1],
                         boxpoints="outliers", marker=dict(size=3, opacity=.3)), 1, 2)
    fig.update_layout(title="Outlier check (1.5 × IQR rule)", showlegend=False)
    p_out = int(df["Price_per_SqFt_Outlier"].sum())
    s_out = int(df["Size_in_SqFt_Outlier"].sum())
    note = (f"{p_out:,} listings ({p_out / len(df):.1%}) are unusually priced per sq ft and "
            f"{s_out:,} are unusually sized. I kept them because they could be genuine luxury or "
            f"bargain homes, but flagged them so they can be filtered out.")
    return fig, note


# ---------------------------------------------------------------- 6-10 location
def q6_ppsf_by_state(df):
    s = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", color_discrete_sequence=[SERIES[0]],
                 labels={"x": "Average price per sq ft (₹)", "y": ""})
    fig.update_layout(title="Average price per sq ft by state", height=560)
    fig.update_yaxes(dtick=1)
    ratio = s.iloc[-1] / s.iloc[0]
    if ratio < 1.1:
        note = (f"{s.index[-1]} tops the list at {rupees(s.iloc[-1])}/sq ft and {s.index[0]} is "
                f"lowest at {rupees(s.iloc[0])}, only {ratio - 1:.1%} apart. Price levels are "
                f"almost flat across states.")
    else:
        note = (f"{s.index[-1]} is the priciest state at {rupees(s.iloc[-1])}/sq ft, "
                f"about {ratio:.1f}× {s.index[0]} ({rupees(s.iloc[0])}/sq ft).")
    return fig, note


def q7_price_by_city(df):
    s = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", color_discrete_sequence=[SERIES[0]],
                 labels={"x": "Average price (₹ lakhs)", "y": ""})
    fig.update_layout(title="Average property price by city", height=max(500, 18 * len(s)))
    fig.update_yaxes(dtick=1)
    top3 = ", ".join(s.index[-3:][::-1])
    gap = s.iloc[-1] / s.iloc[0] - 1
    note = (f"The three most expensive cities are {top3}. The average home in "
            f"{s.index[-1]} costs {money(s.iloc[-1])} vs {money(s.iloc[0])} in {s.index[0]}"
            + (f", a gap of just {gap:.1%}. City matters far less here than it would in a "
               f"real market." if gap < 0.1 else "."))
    return fig, note


def q8_age_by_locality(df):
    s = df.groupby("Locality")["Age_of_Property"].median()
    top = s.sort_values(ascending=False).head(15).sort_values()
    fig = px.bar(x=top.values, y=top.index, orientation="h", color_discrete_sequence=[SERIES[0]],
                 labels={"x": "Median age (years)", "y": ""})
    fig.update_layout(title="Localities with the oldest housing stock (median age)", height=480)
    note = (f"Across {len(s):,} localities the median property age ranges from {s.min():.0f} "
            f"to {s.max():.0f} years; most localities sit around {s.median():.0f}. "
            f"{top.index[-1]} has the oldest stock.")
    return fig, note


def q9_bhk_by_city(df):
    tab = pd.crosstab(df["City"], df["BHK"], normalize="index")
    tab = tab.loc[tab.mean(axis=1).sort_values().index]
    # colour scale starts at 0 so a 19% vs 21% split doesn't look dramatic
    fig = px.imshow(tab, aspect="auto", color_continuous_scale=SEQUENTIAL,
                    zmin=0, zmax=max(0.4, float(tab.values.max())),
                    labels=dict(x="BHK", y="", color="Share"), text_auto=".0%")
    fig.update_layout(title="BHK mix within each city", height=max(500, 18 * len(tab)))
    fig.update_yaxes(dtick=1)
    fig.update_traces(textfont_size=10)
    overall = df["BHK"].value_counts(normalize=True).sort_index()
    common = overall.idxmax()
    spread = tab.max(axis=0) - tab.min(axis=0)
    note = (f"{common} BHK is the most common configuration overall ({_pct(overall.max())}). "
            f"The mix is similar city to city - the biggest gap for any BHK is "
            f"{spread.max():.0%}.")
    return fig, note


def q10_top_locality_trends(df):
    top5 = df.groupby("Locality")["Price_per_SqFt"].mean().nlargest(5).index
    d = df[df["Locality"].isin(top5)].copy()
    # no sale dates in the data, so "trend" = how price moves with construction era
    d["Built"] = (d["Year_Built"] // 5) * 5
    t = d.groupby(["Built", "Locality"])["Price_per_SqFt"].mean().reset_index()
    fig = px.line(t, x="Built", y="Price_per_SqFt", color="Locality", markers=True,
                  color_discrete_sequence=SERIES,
                  labels={"Built": "Built in (5-year bands)", "Price_per_SqFt": "Avg price per sq ft (₹)"})
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(title="Top 5 most expensive localities - price by construction era")
    newer = d[d["Year_Built"] >= 2015]["Price_per_SqFt"].mean()
    older = d[d["Year_Built"] < 2005]["Price_per_SqFt"].mean()
    diff = newer / older - 1
    note = (f"The dataset has no sale dates, so I used year built as the timeline. In these "
            f"five localities, homes built since 2015 average {rupees(newer)}/sq ft vs "
            f"{rupees(older)} for pre-2005 stock"
            + (f", which is no real difference. The zig-zag lines are noise, not a trend."
               if abs(diff) < 0.03 else f", so newer buildings carry a {diff:+.0%} premium."))
    return fig, note


# ---------------------------------------------------------------- 11-15 relationships
CORR_COLS = ["Price_in_Lakhs", "Price_per_SqFt", "Size_in_SqFt", "BHK", "Age_of_Property",
             "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors",
             "Amenity_Count", "Infrastructure_Score", "Expected_Annual_Growth", "Future_Price_5Y"]


def q11_correlation(df):
    c = df[CORR_COLS].corr().round(2)
    fig = px.imshow(c, text_auto=True, zmin=-1, zmax=1, color_continuous_scale=DIVERGING,
                    aspect="auto")
    fig.update_traces(textfont_size=10)
    fig.update_layout(title="How the numeric features move together", height=620)
    pairs = (c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack()
             .rename("r").reset_index())
    pairs = pairs[~pairs.apply(lambda r: {r.level_0, r.level_1} == {"Price_in_Lakhs", "Future_Price_5Y"}, axis=1)]
    top = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(3)
    bits = [f"{a.replace('_', ' ')} & {b.replace('_', ' ')} ({r:+.2f})" for a, b, r in top.values]
    raw = ["Price_in_Lakhs", "Size_in_SqFt", "BHK", "Age_of_Property", "Nearby_Schools",
           "Nearby_Hospitals", "Floor_No", "Total_Floors"]
    rc = c.loc[raw, raw].where(~np.eye(len(raw), dtype=bool)).abs().max().max()
    note = ("Strongest relationships (leaving out current vs future price, which are "
            "naturally tied): " + "; ".join(bits) + ". Most of these are built in: price per "
            "sq ft is price ÷ size, the scores are made from the raw columns, and floor numbers "
            "were capped at the building height during cleaning."
            + (f" Among the original columns, nothing correlates above {rc:.2f}." if rc < 0.1 else ""))
    return fig, note


def _mean_by(df, col, target="Price_per_SqFt"):
    return df.groupby(col)[target].mean()


def q12_schools(df):
    s = _mean_by(df, "Nearby_Schools")
    fig = px.line(x=s.index, y=s.values, markers=True, color_discrete_sequence=[SERIES[0]],
                  labels={"x": "Nearby schools", "y": "Avg price per sq ft (₹)"})
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(title="Nearby schools vs price per sq ft")
    lift = s.iloc[-1] / s.iloc[0] - 1
    r = df["Nearby_Schools"].corr(df["Price_per_SqFt"])
    note = (f"Going from {s.index[0]} to {s.index[-1]} nearby schools changes the average rate "
            f"by {lift:+.1%} (r = {r:.2f}). "
            + ("Families clearly pay for school access." if abs(r) > 0.2 else
               "No visible effect: schools nearby don't move the price in this data." if abs(r) < 0.05 else
               "Families do pay a little for school access, but it's a small effect."))
    return fig, note


def q13_hospitals(df):
    s = _mean_by(df, "Nearby_Hospitals")
    fig = px.line(x=s.index, y=s.values, markers=True, color_discrete_sequence=[SERIES[0]],
                  labels={"x": "Nearby hospitals", "y": "Avg price per sq ft (₹)"})
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(title="Nearby hospitals vs price per sq ft")
    lift = s.iloc[-1] / s.iloc[0] - 1
    r = df["Nearby_Hospitals"].corr(df["Price_per_SqFt"])
    note = (f"More hospitals nearby moves the average rate by {lift:+.1%} from fewest to most "
            f"(r = {r:.2f})" + (". In practice, no effect." if abs(r) < 0.05 else "."))
    return fig, note


def q14_furnished(df):
    order = ["Unfurnished", "Semi-furnished", "Furnished"]
    order = [o for o in order if o in df["Furnished_Status"].unique()] or None
    fig = px.box(df, x="Furnished_Status", y="Price_in_Lakhs", points=False,
                 category_orders={"Furnished_Status": order} if order else None,
                 color_discrete_sequence=[SERIES[0]],
                 labels={"Furnished_Status": "", "Price_in_Lakhs": "Price (₹ lakhs)"})
    fig.update_layout(title="Price by furnishing")
    med = df.groupby("Furnished_Status")["Price_in_Lakhs"].median().sort_values()
    note = (f"Median prices: " + ", ".join(f"{k} {money(v)}" for k, v in med.items()) +
            f". The gap between top and bottom is {med.iloc[-1] / med.iloc[0] - 1:+.1%}"
            + (", so furnishing barely moves the price - buyers mostly price the shell."
               if med.iloc[-1] / med.iloc[0] - 1 < 0.03 else
               ", so furnishing is worth paying attention to when comparing listings."))
    return fig, note


def q15_facing(df):
    s = _mean_by(df, "Facing").sort_values()
    fig = px.bar(x=s.index, y=s.values, color_discrete_sequence=[SERIES[0]],
                 labels={"x": "", "y": "Avg price per sq ft (₹)"})
    fig.update_layout(title="Price per sq ft by facing direction")
    note = (f"{s.index[-1]}-facing homes average {rupees(s.iloc[-1])}/sq ft vs "
            f"{rupees(s.iloc[0])} for {s.index[0].lower()}-facing - a {s.iloc[-1] / s.iloc[0] - 1:.1%} "
            f"difference. "
            + ("Vastu preferences don't show up as a real premium in this data."
               if s.iloc[-1] / s.iloc[0] - 1 < 0.03 else
               "That's a noticeable premium for the preferred direction."))
    return fig, note


# ---------------------------------------------------------------- 16-20 investment
def _count_bar(df, col, title):
    s = df[col].value_counts().sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", color_discrete_sequence=[SERIES[0]],
                 labels={"x": "Listings", "y": ""}, text=s.values)
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, height=300)
    return fig, s


def q16_owner_type(df):
    fig, s = _count_bar(df, "Owner_Type", "Listings by owner type")
    note = (f"{s.index[-1]}s list the most properties ({s.iloc[-1]:,}, "
            f"{s.iloc[-1] / s.sum():.0%}); {s.index[0].lower()}s the fewest ({s.iloc[0]:,}).")
    return fig, note


def q17_availability(df):
    fig, s = _count_bar(df, "Availability_Status", "Listings by availability")
    labels = {k: k.replace("_", " ") for k in s.index}
    note = ", ".join(f"{labels[k]}: {v:,} ({v / s.sum():.0%})" for k, v in s[::-1].items()) + "."
    return fig, note


def q18_parking(df):
    d = df.assign(Parking=df["Parking_Space"].map({1: "Has parking", 0: "No parking"}))
    med = d.groupby("Parking")["Price_in_Lakhs"].median()
    ppsf = d.groupby("Parking")["Price_per_SqFt"].median()
    fig = px.box(d, x="Parking", y="Price_in_Lakhs", points=False, color_discrete_sequence=[SERIES[0]],
                 labels={"Parking": "", "Price_in_Lakhs": "Price (₹ lakhs)"})
    fig.update_layout(title="Does parking change the price?")
    diff = med.get("Has parking", np.nan) / med.get("No parking", np.nan) - 1
    note = (f"Median price with parking is {money(med.get('Has parking', 0))} vs "
            f"{money(med.get('No parking', 0))} without ({diff:+.1%}); per sq ft the gap is "
            f"{ppsf.get('Has parking', 0) / ppsf.get('No parking', 1) - 1:+.1%}.")
    return fig, note


def q19_amenities(df):
    rows = []
    for a in ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]:
        col = f"Has_{a}"
        if col in df:
            m = df.groupby(col)["Price_per_SqFt"].mean()
            rows.append((a, m.get(1, np.nan) / m.get(0, np.nan) - 1))
    lift = pd.DataFrame(rows, columns=["Amenity", "Lift"]).sort_values("Lift")
    fig = px.bar(lift, x="Lift", y="Amenity", orientation="h", color_discrete_sequence=[SERIES[0]],
                 labels={"Lift": "Price per sq ft vs homes without it", "Amenity": ""})
    fig.update_xaxes(tickformat="+.1%")
    fig.update_layout(title="Price premium attached to each amenity", height=320)
    by_count = df.groupby("Amenity_Count")["Price_per_SqFt"].mean()
    best = lift.iloc[-1]
    lead = (f"{best['Amenity']} carries the biggest premium ({best['Lift']:+.1%}). "
            if abs(best["Lift"]) >= 0.01 else
            "No single amenity moves the price per sq ft by even 1% - buyers don't seem to "
            "pay extra for a gym or pool on its own. ")
    note = (lead + f"Homes with {by_count.index[-1]} amenities average "
            f"{rupees(by_count.iloc[-1])}/sq ft vs {rupees(by_count.iloc[0])} with "
            f"{by_count.index[0]}.")
    return fig, note


def q20_transport(df):
    order = ["Low", "Medium", "High"]
    g = df.groupby("Public_Transport_Accessibility").agg(
        ppsf=("Price_per_SqFt", "mean"), good=("Good_Investment", "mean")).reindex(order)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg price per sq ft (₹)",
                                                        "Share rated Good Investment"))
    fig.add_trace(go.Bar(x=g.index, y=g["ppsf"], marker_color=SERIES[0]), 1, 1)
    fig.add_trace(go.Bar(x=g.index, y=g["good"], marker_color=GOOD), 1, 2)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_layout(title="Public transport access", showlegend=False)
    diff = g.loc['High', 'ppsf'] / g.loc['Low', 'ppsf'] - 1
    note = ((f"Today's price per sq ft is about the same whatever the transport access "
             f"({diff:+.1%} high vs low). " if abs(diff) < 0.03 else
             f"High-access homes cost {diff:+.1%} per sq ft compared with low-access ones. ")
            + f"Investment potential is where it shows: {g.loc['High', 'good']:.0%} of high-access "
            f"homes qualify as good investments vs {g.loc['Low', 'good']:.0%} of low-access ones, "
            f"because connectivity is one of the drivers of expected growth.")
    return fig, note


# ---------------------------------------------------------------- Step 2 extras
def x_city_trends(df):
    d = df.copy()
    top = d.groupby("City")["Price_per_SqFt"].median().nlargest(6).index
    d = d[d["City"].isin(top)]
    d["Built"] = (d["Year_Built"] // 5) * 5
    t = d.groupby(["Built", "City"])["Price_per_SqFt"].median().reset_index()
    fig = px.line(t, x="Built", y="Price_per_SqFt", color="City", markers=True,
                  color_discrete_sequence=SERIES,
                  labels={"Built": "Built in (5-year bands)", "Price_per_SqFt": "Median ₹ per sq ft"})
    fig.update_traces(line_width=2, marker_size=7)
    fig.update_layout(title="Price trends in the six priciest cities")
    leader = t.loc[t.groupby("Built")["Price_per_SqFt"].idxmax(), "City"]
    newer_wins = all(g.sort_values("Built")["Price_per_SqFt"].iloc[-1] >
                     g.sort_values("Built")["Price_per_SqFt"].iloc[0] for _, g in t.groupby("City"))
    note = ((f"{top[0]} stays on top across every construction era. " if leader.nunique() == 1
             else f"{leader.value_counts().index[0]} leads in most construction eras. ")
            + ("Newer stock is priced higher in every one of these cities - age discounts the price."
               if newer_wins else
               "Newer stock isn't consistently pricier, so build year tells you little about price here."))
    return fig, note


def x_area_vs_return(df):
    d = df.copy()
    d["Size band"] = pd.cut(d["Size_in_SqFt"], [0, 1000, 1500, 2000, 3000, 6000],
                            labels=["<1,000", "1,000-1,500", "1,500-2,000", "2,000-3,000", "3,000+"])
    g = d.groupby("Size band", observed=True)["Expected_Annual_Growth"].mean()
    fig = px.bar(x=g.index.astype(str), y=g.values, color_discrete_sequence=[SERIES[0]],
                 labels={"x": "Size (sq ft)", "y": "Avg expected growth / yr"})
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(title="Area vs expected investment return")
    r = df["Size_in_SqFt"].corr(df["Expected_Annual_Growth"])
    best = g.idxmax()
    note = (f"Correlation between size and expected return is {r:.2f}. "
            + ("Size drives the price tag, not the growth rate. " if abs(r) < 0.2 else "")
            + f"The {best} sq ft band has the best expected return ({g.max():.2%} a year)."
            + (" Bigger homes come out ahead because asking prices here don't rise with size, "
               "so a big home usually has a low price per sq ft, which counts as being priced "
               "below its local market." if r > 0.2 else ""))
    return fig, note


def x_security(df):
    g = df.groupby("Security")["Good_Investment"].mean().rename({1: "Secured", 0: "No security"})
    fig = px.bar(x=g.index, y=g.values, color_discrete_sequence=[GOOD],
                 labels={"x": "", "y": "Share rated Good Investment"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Security (crime-risk proxy) vs Good Investment")
    note = ("The dataset has no crime-rate column, so I used the Security flag (gated/CCTV/"
            f"guard) as the closest proxy. {g.get('Secured', 0):.0%} of secured properties are "
            f"good investments vs {g.get('No security', 0):.0%} without.")
    return fig, note


def x_infra_vs_resale(df):
    d = df.copy()
    d["Infra band"] = pd.cut(d["Infrastructure_Score"], [-0.1, 2, 4, 6, 8, 10],
                             labels=["0-2", "2-4", "4-6", "6-8", "8-10"])
    g = d.groupby("Infra band", observed=True).apply(
        lambda x: (x["Future_Price_5Y"] / x["Price_in_Lakhs"]).mean() - 1, include_groups=False)
    fig = px.bar(x=g.index.astype(str), y=g.values, color_discrete_sequence=[SERIES[0]],
                 labels={"x": "Infrastructure score", "y": "Expected 5-year value gain"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Infrastructure score vs 5-year resale value")
    note = (f"Properties in the best-served areas (8-10) are expected to gain "
            f"{g.iloc[-1]:.0%} in five years vs {g.iloc[0]:.0%} in the weakest - roughly "
            f"{(g.iloc[-1] - g.iloc[0]) * 100:.0f} extra points of growth.")
    return fig, note


QUESTIONS = [
    ("Price & size", 1, "What is the distribution of property prices?", q1_price_distribution),
    ("Price & size", 2, "What is the distribution of property sizes?", q2_size_distribution),
    ("Price & size", 3, "How does price per sq ft vary by property type?", q3_ppsf_by_type),
    ("Price & size", 4, "Is there a relationship between property size and price?", q4_size_vs_price),
    ("Price & size", 5, "Are there outliers in price per sq ft or size?", q5_outliers),
    ("Location", 6, "What is the average price per sq ft by state?", q6_ppsf_by_state),
    ("Location", 7, "What is the average property price by city?", q7_price_by_city),
    ("Location", 8, "What is the median age of properties by locality?", q8_age_by_locality),
    ("Location", 9, "How is BHK distributed across cities?", q9_bhk_by_city),
    ("Location", 10, "What are the price trends for the top 5 most expensive localities?", q10_top_locality_trends),
    ("Relationships", 11, "How are numeric features correlated with each other?", q11_correlation),
    ("Relationships", 12, "How do nearby schools relate to price per sq ft?", q12_schools),
    ("Relationships", 13, "How do nearby hospitals relate to price per sq ft?", q13_hospitals),
    ("Relationships", 14, "How does price vary by furnished status?", q14_furnished),
    ("Relationships", 15, "How does price per sq ft vary by facing direction?", q15_facing),
    ("Investment & amenities", 16, "How many properties belong to each owner type?", q16_owner_type),
    ("Investment & amenities", 17, "How many properties are available under each status?", q17_availability),
    ("Investment & amenities", 18, "Does parking space affect property price?", q18_parking),
    ("Investment & amenities", 19, "How do amenities affect price per sq ft?", q19_amenities),
    ("Investment & amenities", 20, "How does public transport relate to price and investment potential?", q20_transport),
]

EXTRAS = [
    ("Price trends by city", x_city_trends),
    ("Area vs investment return", x_area_vs_return),
    ("Crime-risk proxy vs Good Investment", x_security),
    ("Infrastructure vs resale value", x_infra_vs_resale),
]
