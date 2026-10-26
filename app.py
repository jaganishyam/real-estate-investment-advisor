"""
Real Estate Investment Advisor - Streamlit app.

    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")

from utils import style  # noqa: E402

style.apply()

pages = [
    st.Page("app_pages/home.py", title="Overview", icon=":material/location_city:", default=True),
    st.Page("app_pages/advisor.py", title="Investment advisor", icon=":material/real_estate_agent:"),
    st.Page("app_pages/explore.py", title="Find properties", icon=":material/travel_explore:"),
    st.Page("app_pages/market.py", title="Market map", icon=":material/map:"),
    st.Page("app_pages/questions.py", title="20 questions (EDA)", icon=":material/query_stats:"),
    st.Page("app_pages/models.py", title="Model lab", icon=":material/model_training:"),
    st.Page("app_pages/method.py", title="How it works", icon=":material/menu_book:"),
]

# my own sidebar menu, so the brand can sit above it
nav = st.navigation(pages, position="hidden")

with st.sidebar:
    st.markdown(style.BRAND, unsafe_allow_html=True)
    for page in pages:
        st.page_link(page)
    st.divider()
    st.caption("Built by Shyam Jagani · Python, scikit-learn, XGBoost, MLflow, Streamlit")

nav.run()
