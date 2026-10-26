import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import AMENITY_LIST, REFERENCE_YEAR, FORECAST_YEARS
from plot_theme import GOOD, CAUTION, MUTED, SERIES, money, rupees
from utils import style
from utils.anim import verdict_card, show
from utils.data import load_data, load_models, models_ready, predict, explain, pretty

style.page_header(
    "Investment advisor",
    "Describe a property and I'll tell you whether it looks like a good investment, what it "
    f"could be worth in {REFERENCE_YEAR + FORECAST_YEARS}, and which details drove the call.")

if not models_ready():
    st.error("No trained models found. Run `python src/train.py` first, then reload this page.")
    st.stop()

df = load_data()
_, _, meta = load_models()
opts = meta["form_options"]

# ------------------------------------------------------------------ examples
EXAMPLES = {
    "Fairly priced 2 BHK, good transport": dict(
        state="Maharashtra", city="Pune", ptype="Apartment", bhk=2, size=1050, price=72.0,
        year=2019, furn="Semi-furnished", floor=6, floors=14, facing="East", parking=True,
        security=True, amen=["Gym", "Garden", "Playground"], owner="Builder",
        avail="Under_Construction", schools=8, hospitals=7, transport="High"),
    "Pricey old villa, poor connectivity": dict(
        state="Delhi", city="New Delhi", ptype="Villa", bhk=5, size=3800, price=420.0,
        year=1996, furn="Furnished", floor=1, floors=2, facing="North", parking=True,
        security=False, amen=["Garden"], owner="Broker", avail="Ready_to_Move",
        schools=3, hospitals=2, transport="Low"),
}

DEFAULT = EXAMPLES["Fairly priced 2 BHK, good transport"]
KEYS = ["state", "city", "ptype", "bhk", "size", "price", "year", "furn", "floor", "floors",
        "facing", "parking", "security", "amen", "owner", "avail", "schools", "hospitals", "transport"]


def _safe(value, options):
    return value if value in options else options[0]


def load_example(vals: dict):
    """Write an example straight into the widget keys so every field updates."""
    checks = {"state": opts["State"], "ptype": opts["Property_Type"], "furn": opts["Furnished_Status"],
              "facing": opts["Facing"], "owner": opts["Owner_Type"], "avail": opts["Availability_Status"],
              "transport": opts["Public_Transport_Accessibility"]}
    for k in KEYS:
        val = vals[k]
        if k in checks:
            val = _safe(val, checks[k])
        if k == "price":
            val = float(val)
        if k == "amen":
            val = [a for a in val if a in AMENITY_LIST]
        st.session_state[f"f_{k}"] = val


if "f_state" not in st.session_state:
    load_example(DEFAULT)

st.caption("Not sure what to enter? Load an example:")
ex_cols = st.columns(len(EXAMPLES) + 2)
for col, (label, vals) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, width="stretch"):
        load_example(vals)
        st.session_state.pop("result", None)
        st.rerun()

# state/city live outside the form so the city list follows the state
loc1, loc2, _ = st.columns([1, 1, 1])
state = loc1.selectbox("State", opts["State"], key="f_state")
cities = opts["state_city"].get(state, opts["City"])
if st.session_state.get("f_city") not in cities:
    st.session_state.f_city = cities[0]
city = loc2.selectbox("City", cities, key="f_city")

with st.form("advisor_form", border=False):
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1.container(border=True):
        st.markdown("### The home")
        ptype = st.selectbox("Property type", opts["Property_Type"], key="f_ptype")
        bhk = st.select_slider("BHK", options=[1, 2, 3, 4, 5], key="f_bhk")
        size = st.number_input("Size (sq ft)", 300, 10000, step=50, key="f_size")
        price = st.number_input("Asking price (₹ lakhs)", 5.0, 2000.0, step=1.0, key="f_price")
        year = st.number_input("Year built", 1950, REFERENCE_YEAR, key="f_year")
        furn = st.selectbox("Furnishing", opts["Furnished_Status"], key="f_furn")

    with c2.container(border=True):
        st.markdown("### The building")
        fl1, fl2 = st.columns(2)
        floor = fl1.number_input("Floor", 0, 100, key="f_floor")
        floors = fl2.number_input("Total floors", 1, 100, key="f_floors")
        facing = st.selectbox("Facing", opts["Facing"], key="f_facing")
        amen = st.multiselect("Amenities", AMENITY_LIST, key="f_amen")
        t1, t2 = st.columns(2)
        parking = t1.toggle("Parking", key="f_parking")
        security = t2.toggle("Security", key="f_security")
        owner = st.selectbox("Sold by", opts["Owner_Type"], key="f_owner")
        avail = st.selectbox("Availability", opts["Availability_Status"],
                             format_func=lambda s: s.replace("_", " "), key="f_avail")

    with c3.container(border=True):
        st.markdown("### The neighbourhood")
        schools = st.slider("Schools nearby", 1, 10, key="f_schools")
        hospitals = st.slider("Hospitals nearby", 1, 10, key="f_hospitals")
        transports = [t for t in ["Low", "Medium", "High"] if t in opts["Public_Transport_Accessibility"]]
        transport = st.radio("Public transport access", transports, horizontal=True, key="f_transport")
        st.markdown('<p class="note">Schools and hospitals are counted within the locality, '
                    'on the same 1-10 scale as the dataset.</p>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Assess this property", type="primary")

if submitted:
    form = {
        "State": state, "City": city, "Locality": "", "Property_Type": ptype, "BHK": int(bhk),
        "Size_in_SqFt": int(size), "Price_in_Lakhs": float(price), "Year_Built": int(year),
        "Furnished_Status": furn, "Floor_No": int(min(floor, floors)), "Total_Floors": int(floors),
        "Nearby_Schools": int(schools), "Nearby_Hospitals": int(hospitals),
        "Public_Transport_Accessibility": transport, "Parking_Space": int(parking),
        "Security": int(security), "Amenities_list": amen, "Facing": facing,
        "Owner_Type": owner, "Availability_Status": avail,
    }
    st.session_state.result = predict(form)

res = st.session_state.get("result")
if res is None:
    st.info("Fill in the details above and press **Assess this property**.")
    st.stop()

# ------------------------------------------------------------------ verdict
row = res["row"].iloc[0]
st.markdown("## The verdict")
html, h = verdict_card(good=res["good"], prob=res["prob"], price_now=res["price_now"],
                       price_5y=res["price_5y"], low_5y=res["low_5y"], high_5y=res["high_5y"],
                       cagr=res["cagr"], start_year=REFERENCE_YEAR, city=row["City"])
show(html, h)

gain = res["price_5y"] - res["price_now"]
st.markdown(
    f'<p class="note">Models used: <b>{meta["classification"]["best"]}</b> for the verdict and '
    f'<b>{meta["regression"]["best"]}</b> for the price. Expected gain over five years: '
    f'<b>{money(gain)}</b>. The likely range covers 90% of the model\'s errors on held-out listings. '
    f'This is a model estimate for a course project, not financial advice.</p>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ why
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("## What drove the verdict")
    contrib = explain(res["X"])
    if contrib is not None:
        top = contrib.head(8)[::-1]
        colors = [GOOD if x > 0 else CAUTION for x in top.values]
        fig = go.Figure(go.Bar(
            x=top.values, y=[pretty(i) for i in top.index], orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.add_vline(0, line_color=MUTED, line_width=1)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=40),
                          xaxis_title="← pushes towards ‘think twice’   ·   pushes towards ‘good’ →")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        ups = [pretty(i) for i in contrib[contrib > 0].head(2).index]
        downs = [pretty(i) for i in contrib[contrib < 0].head(2).index]
        bits = []
        if ups:
            bits.append(f"Working in its favour: <b>{', '.join(ups).lower()}</b>.")
        if downs:
            bits.append(f"Holding it back: <b>{', '.join(downs).lower()}</b>.")
        style.takeaway(" ".join(bits))
    else:
        imp = next(r["importance"] for r in meta["classification"]["results"]
                   if r["model"] == meta["classification"]["best"])
        st.bar_chart(dict(list(imp.items())[:8]))

with right:
    st.markdown(f"## How it compares in {row['City']}")
    city_df = df[df["City"] == row["City"]]
    ppsf = row["Price_per_SqFt"]
    pct = (city_df["Price_per_SqFt"] < ppsf).mean()
    fig = px.histogram(city_df, x="Price_per_SqFt", nbins=50, color_discrete_sequence=[SERIES[0]],
                       labels={"Price_per_SqFt": "Price per sq ft (₹)"})
    fig.update_traces(opacity=.55, hovertemplate="₹%{x:,.0f}: %{y:,} listings<extra></extra>")
    fig.add_vline(ppsf, line_color=SERIES[1], line_width=2.5,
                  annotation_text="this property", annotation_font_color=SERIES[1])
    fig.update_layout(height=300, yaxis_title="Listings", margin=dict(l=10, r=10, t=30, b=40))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    style.takeaway(
        f"At {rupees(ppsf)}/sq ft it's "
        + (f"pricier than {pct:.0%} of {row['City']} listings " if pct >= 0.5
           else f"cheaper than {1 - pct:.0%} of {row['City']} listings ")
        + f"(city median {rupees(row['City_Median_PPSF'])}). Infrastructure score "
        f"{row['Infrastructure_Score']:.1f}/10.")

# ------------------------------------------------------------------ similar
st.markdown("## Similar listings")
sim = df[(df["City"] == row["City"]) & (df["BHK"] == row["BHK"])
         & df["Size_in_SqFt"].between(row["Size_in_SqFt"] * .8, row["Size_in_SqFt"] * 1.2)]
if sim.empty:
    st.caption("No listings in the data are close enough to compare.")
else:
    style.stat_strip([
        (f"{len(sim):,}", f"{int(row['BHK'])} BHK listings of a similar size in {row['City']}"),
        (money(sim["Price_in_Lakhs"].median()), "Their median price"),
        (f"{sim['Good_Investment'].mean():.0%}", "Rated good investments"),
    ])
    show = (sim.sort_values("Expected_Annual_Growth", ascending=False).head(10)
            [["Locality", "Property_Type", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
              "Infrastructure_Score", "Expected_Annual_Growth", "Future_Price_5Y", "Good_Investment"]])
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "Property_Type": "Type",
            "Size_in_SqFt": st.column_config.NumberColumn("Size (sq ft)", format="%d"),
            "Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
            "Price_per_SqFt": st.column_config.NumberColumn("₹ / sq ft", format="%d"),
            "Infrastructure_Score": st.column_config.ProgressColumn("Infrastructure", min_value=0, max_value=10, format="%.1f"),
            "Expected_Annual_Growth": st.column_config.NumberColumn("Growth / yr", format="percent"),
            "Future_Price_5Y": st.column_config.NumberColumn("In 5 yrs (₹ L)", format="%.1f"),
            "Good_Investment": st.column_config.CheckboxColumn("Good buy"),
        })
    st.caption("The ten similar listings with the strongest expected growth.")
