import plotly.express as px
import streamlit as st

from plot_theme import GOOD, SERIES, money, rupees
from utils import style
from utils.data import load_data

style.page_header("Find properties",
                  "Narrow the listings down by location, budget, size and BHK. The table shows "
                  "each match's expected growth and whether it's rated a good buy.")

df = load_data()

# ------------------------------------------------------------------ filters
with st.container(border=True):
    f1, f2, f3 = st.columns([1.4, 1, 1], gap="medium")
    cities = f1.multiselect("Cities", sorted(df["City"].unique()), placeholder="All cities")
    bhk = f2.pills("BHK", sorted(df["BHK"].unique()), selection_mode="multi", default=None)
    ptype = f3.pills("Property type", sorted(df["Property_Type"].unique()), selection_mode="multi")

    f4, f5, f6 = st.columns([1.4, 1, 1], gap="medium")
    pmin, pmax = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
    price = f4.slider("Budget (₹ lakhs)", pmin, pmax, (pmin, pmax), step=5.0)
    smin, smax = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
    size = f5.slider("Size (sq ft)", smin, smax, (smin, smax), step=50)
    with f6:
        st.write("")
        only_good = st.toggle("Only good investments")
        ready = st.toggle("Ready to move only")
        no_outliers = st.toggle("Hide unusual prices", help="Drops listings flagged as outliers on price per sq ft.")

m = df["Price_in_Lakhs"].between(*price) & df["Size_in_SqFt"].between(*size)
if cities:
    m &= df["City"].isin(cities)
if bhk:
    m &= df["BHK"].isin(bhk)
if ptype:
    m &= df["Property_Type"].isin(ptype)
if only_good:
    m &= df["Good_Investment"] == 1
if ready:
    m &= df["Is_Ready_To_Move"] == 1
if no_outliers:
    m &= df["Price_per_SqFt_Outlier"] == 0
res = df[m]

if res.empty:
    st.warning("Nothing matches these filters. Try widening the budget or size range, or removing a city.")
    st.stop()

style.stat_strip([
    (f"{len(res):,}", "Matching listings"),
    (money(res["Price_in_Lakhs"].median()), "Median price"),
    (rupees(res["Price_per_SqFt"].median()), "Median price per sq ft"),
    (f"{res['Good_Investment'].mean():.0%}", "Rated good investments"),
    (f"{res['Expected_Annual_Growth'].mean():.1%}", "Average expected growth a year"),
])

# ------------------------------------------------------------------ results
sort_opts = {
    "Highest expected growth": ("Expected_Annual_Growth", False),
    "Lowest price": ("Price_in_Lakhs", True),
    "Best value (₹ per sq ft vs city)": ("Price_vs_City", True),
    "Largest": ("Size_in_SqFt", False),
    "Newest": ("Year_Built", False),
}
s1, s2 = st.columns([1, 2])
sort_by = s1.selectbox("Sort by", list(sort_opts))
col, asc = sort_opts[sort_by]
table = res.sort_values(col, ascending=asc).head(500)

st.dataframe(
    table[["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
           "Price_per_SqFt", "Year_Built", "Availability_Status", "Infrastructure_Score",
           "Expected_Annual_Growth", "Future_Price_5Y", "Good_Investment"]],
    hide_index=True, width="stretch", height=420,
    column_config={
        "Property_Type": "Type",
        "Size_in_SqFt": st.column_config.NumberColumn("Size (sq ft)", format="%d"),
        "Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
        "Price_per_SqFt": st.column_config.NumberColumn("₹ / sq ft", format="%d"),
        "Year_Built": st.column_config.NumberColumn("Built", format="%d"),
        "Availability_Status": "Status",
        "Infrastructure_Score": st.column_config.ProgressColumn("Infrastructure", min_value=0, max_value=10, format="%.1f"),
        "Expected_Annual_Growth": st.column_config.NumberColumn("Growth / yr", format="percent"),
        "Future_Price_5Y": st.column_config.NumberColumn("In 5 yrs (₹ L)", format="%.1f"),
        "Good_Investment": st.column_config.CheckboxColumn("Good buy"),
    })
st.caption(f"Showing the first {len(table):,} of {len(res):,} matches.")
st.download_button("Download all matches (CSV)", res.to_csv(index=False).encode(),
                   file_name="matching_properties.csv", mime="text/csv")

# ------------------------------------------------------------------ quick read
st.markdown("## What these matches look like")
a, b = st.columns(2, gap="large")
with a:
    by_city = res.groupby("City").agg(n=("ID", "size"), good=("Good_Investment", "mean"))
    by_city = by_city.sort_values("n").tail(12)
    fig = px.bar(by_city, x="n", y=by_city.index, orientation="h", color_discrete_sequence=[SERIES[0]],
                 labels={"n": "Matching listings", "y": ""})
    fig.update_traces(hovertemplate="%{y}: %{x:,} listings<extra></extra>")
    fig.update_layout(height=380, title="Where they are", yaxis_title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with b:
    d = res.sample(min(4000, len(res)), random_state=1).copy()
    d["Verdict"] = d["Good_Investment"].map({1: "Good investment", 0: "Think twice"})
    fig = px.scatter(d, x="Size_in_SqFt", y="Price_in_Lakhs", color="Verdict", opacity=.5,
                     color_discrete_map={"Good investment": GOOD, "Think twice": "#B9BFD6"},
                     labels={"Size_in_SqFt": "Size (sq ft)", "Price_in_Lakhs": "Price (₹ lakhs)"},
                     category_orders={"Verdict": ["Good investment", "Think twice"]})
    fig.update_traces(marker_size=5)
    fig.update_layout(height=380, title="Price vs size, coloured by verdict", legend_title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
