import plotly.express as px
import streamlit as st

from config import REFERENCE_YEAR, FORECAST_YEARS
from plot_theme import GOOD, money
from utils import style
from utils.anim import skyline_hero, show
from utils.data import load_data

df = load_data()

n_cities = df["City"].nunique()
good_share = df["Good_Investment"].mean()
html, h = skyline_hero(
    headline="Know what a home could be worth in five years, before you buy it.",
    sub=(f"I trained two models on {len(df):,} listings from {n_cities} Indian cities. One says "
         f"whether a property is a good investment, the other estimates its price in "
         f"{REFERENCE_YEAR + FORECAST_YEARS}."),
    chips=[(f"{len(df) / 1000:.0f}k", "listings"), (str(n_cities), "cities"),
           (f"{good_share:.0%}", "rated good investments")],
)
show(html, h)

st.write("")
style.stat_strip([
    (money(df["Price_in_Lakhs"].median()), "Median asking price"),
    (f"₹{df['Price_per_SqFt'].median():,.0f}", "Median price per sq ft"),
    (f"{df['Size_in_SqFt'].median():,.0f} sq ft", "Typical home size"),
    (f"{df['Expected_Annual_Growth'].mean():.1%}", "Average expected growth a year"),
    (money(df["Future_Price_5Y"].median()), "Median value in 5 years"),
])

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("## Where to start")
    st.page_link("app_pages/advisor.py", label="Check a property you're considering",
                 icon=":material/real_estate_agent:")
    st.caption("Enter its details and get a verdict, a 5-year price and the reasons behind both.")
    st.page_link("app_pages/explore.py", label="Browse listings by budget, size and BHK",
                 icon=":material/travel_explore:")
    st.caption("Filter 250k listings down to the ones that fit, and see which are rated good buys.")
    st.page_link("app_pages/market.py", label="Compare cities on the map", icon=":material/map:")
    st.caption("Price levels, growth outlook and good-investment share for every city.")
    st.page_link("app_pages/questions.py", label="Read the 20-question analysis",
                 icon=":material/query_stats:")
    st.caption("Every question from the brief, answered with a chart and a plain-English takeaway.")

with right:
    st.markdown("## Where the good buys are")
    g = (df.groupby("City")["Good_Investment"].mean().sort_values(ascending=True).tail(12))
    fig = px.bar(x=g.values, y=g.index, orientation="h", color_discrete_sequence=[GOOD],
                 labels={"x": "Share of listings rated good investments", "y": ""})
    fig.update_xaxes(tickformat=".0%")
    fig.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Top 12 cities by share of listings that clear both the growth and pricing bars.")
