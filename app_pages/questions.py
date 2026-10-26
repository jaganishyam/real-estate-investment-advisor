import streamlit as st

from eda_questions import QUESTIONS
from utils import style
from utils.data import load_data

style.page_header("20 questions (EDA)",
                  "Every question from the project brief, answered with a chart and what I "
                  "took away from it. The takeaways are calculated from the data, not typed in.")

df = load_data()


@st.cache_resource(show_spinner="Drawing charts…")
def answer(num: int):
    fn = next(f for _, n, _, f in QUESTIONS if n == num)
    return fn(df)


groups = list(dict.fromkeys(g for g, *_ in QUESTIONS))
labels = {g: f"{g} ({min(n for gg, n, *_ in QUESTIONS if gg == g)}-"
             f"{max(n for gg, n, *_ in QUESTIONS if gg == g)})" for g in groups}
pick = st.segmented_control("Section", groups, default=groups[0], format_func=labels.get)
pick = pick or groups[0]

for group, num, question, _ in QUESTIONS:
    if group != pick:
        continue
    st.markdown(f'<div style="margin-top:1.6rem"><span class="qnum">{num}</span>'
                f'<span class="qtitle">{question}</span></div>', unsafe_allow_html=True)
    fig, note = answer(num)
    fig.update_layout(title=None, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                    key=f"q{num}")
    style.takeaway(note)
