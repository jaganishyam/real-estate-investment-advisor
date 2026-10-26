import plotly.express as px
import streamlit as st

from eda_questions import EXTRAS
from plot_theme import SEQUENTIAL, money
from utils import style
from utils.data import CITY_COORDS, load_data

style.page_header("Market map",
                  "How price levels and the 5-year outlook change from city to city, plus the "
                  "four location and infrastructure checks the brief asks for.")

df = load_data()

city = (df.groupby(["State", "City"])
        .agg(listings=("ID", "size"), ppsf=("Price_per_SqFt", "median"),
             price=("Price_in_Lakhs", "median"), growth=("Expected_Annual_Growth", "mean"),
             good=("Good_Investment", "mean"))
        .reset_index())
city["lat"] = city["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
city["lon"] = city["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
mapped = city.dropna(subset=["lat"])

metric = st.segmented_control(
    "Colour cities by",
    ["Price per sq ft", "Expected growth", "Good-investment share"],
    default="Price per sq ft")
col, fmt = {"Price per sq ft": ("ppsf", ":,.0f"), "Expected growth": ("growth", ":.2%"),
            "Good-investment share": ("good", ":.0%")}[metric or "Price per sq ft"]

left, right = st.columns([1.25, 1], gap="large")
with left:
    fig = px.scatter_map(
        mapped, lat="lat", lon="lon", size="listings", color=col, hover_name="City",
        color_continuous_scale=SEQUENTIAL[1:], size_max=26, zoom=3.6,
        center={"lat": 22.5, "lon": 80}, map_style="carto-positron",
        hover_data={"lat": False, "lon": False, "listings": ":,", "ppsf": ":,.0f",
                    "growth": ":.2%", "good": ":.0%"},
        labels={"ppsf": "₹ / sq ft", "growth": "Growth / yr", "good": "Good buys", "listings": "Listings"})
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title=None, thickness=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": True})
    missing = len(city) - len(mapped)
    st.caption("Bubble size = number of listings. Scroll to zoom."
               + (f" {missing} cities aren't on the map (no coordinates)." if missing else ""))

with right:
    st.markdown(f"### Cities ranked by {metric.lower() if metric else 'price per sq ft'}")
    st.dataframe(
        city.sort_values(col, ascending=False)[["City", "State", "ppsf", "price", "growth", "good"]],
        hide_index=True, width="stretch", height=520,
        column_config={
            "ppsf": st.column_config.NumberColumn("₹ / sq ft", format="%d"),
            "price": st.column_config.NumberColumn("Median (₹ L)", format="%.1f"),
            "growth": st.column_config.NumberColumn("Growth / yr", format="percent"),
            "good": st.column_config.ProgressColumn("Good buys", min_value=0, max_value=1, format="percent"),
        })

# ------------------------------------------------------------------ heatmap
st.markdown("## Price per sq ft by state and property type")
heat = df.pivot_table(index="State", columns="Property_Type", values="Price_per_SqFt", aggfunc="median")
heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]
fig = px.imshow(heat, text_auto=",.0f", aspect="auto", color_continuous_scale=SEQUENTIAL,
                labels=dict(color="₹ / sq ft", x="", y=""))
fig.update_traces(hovertemplate="%{y} · %{x}: ₹%{z:,.0f}/sq ft<extra></extra>", textfont_size=11)
fig.update_layout(height=max(420, 24 * len(heat)), margin=dict(l=10, r=10, t=10, b=10))
fig.update_yaxes(dtick=1)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
top_state = heat.mean(axis=1).idxmax()
leads_all = all(heat[c].idxmax() == top_state for c in heat.columns)
type_order_same = (heat.rank(axis=1).nunique() == 1).all()
style.takeaway(
    f"{top_state} is the most expensive state "
    + ("for every property type. " if leads_all else "on average, though not for every property type. ")
    + ("The ranking of property types is the same in every state, so location sets the price "
       "level and type adds a premium on top." if type_order_same else
       "Which property type costs most per sq ft varies from state to state.")
    + (f" Keep the scale in mind: the whole grid spans only {heat.max().max() / heat.min().min() - 1:.0%} "
       f"from cheapest to dearest cell, so the colour differences are small in rupee terms."
       if heat.max().max() / heat.min().min() - 1 < 0.15 else ""))

# ------------------------------------------------------------------ step 2 checks
st.markdown("## The four checks from the brief")
tabs = st.tabs([t for t, _ in EXTRAS])
for tab, (title, fn) in zip(tabs, EXTRAS):
    with tab:
        fig, note = fn(df)
        fig.update_layout(height=420, title=None)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        style.takeaway(note)
