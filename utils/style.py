"""
Page styling. Streamlit's defaults are fine for a notebook-style tool, but
this is meant to feel like a property advisor, so: a heavier display face
for headings, a calm paper background, and colour reserved for meaning
(green = good investment, brick = think twice, amber = the lit windows).
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --ink: #1A1F36;
  --night: #1B2240;
  --paper: #F5F6FA;
  --card: #FFFFFF;
  --line: #E3E6EF;
  --muted: #5B6382;
  --amber: #F2B544;
  --good: #1F8A70;
  --good-soft: #E3F3EE;
  --caution: #C2413B;
  --caution-soft: #F8E6E4;
  --blue: #2a78d6;
}

h1, h2, h3, h4, .display {
  font-family: 'Bricolage Grotesque', 'Source Sans 3', sans-serif !important;
  letter-spacing: -0.015em;
  color: var(--ink);
}
h1 { font-weight: 800 !important; font-size: 2.35rem !important; line-height: 1.1 !important; }
h2 { font-weight: 700 !important; font-size: 1.55rem !important; }
h3 { font-weight: 700 !important; font-size: 1.15rem !important; }

.block-container { padding-top: 3.4rem; padding-bottom: 4rem; max-width: 1240px; }

/* sidebar: the "night" colour from the hero, so the two feel like one place */
[data-testid="stSidebar"] { background: var(--night); }
[data-testid="stSidebar"] * { color: #D9DCEB; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a { padding: .45rem .7rem; border-radius: 8px; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: rgba(255,255,255,0.06); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] { background: rgba(242,181,68,0.14); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] *,
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] * { color: var(--amber) !important; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stPageLink"] p { font-size: 1rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }

.brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem; }
.brand svg { flex: 0 0 auto; }
.brand b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.15rem; color:#fff; line-height:1.1; }
.brand small { display:block; color:#9CA3C4; font-size:.8rem; }

/* page intro under each h1 */
.lede { color: var(--muted); font-size: 1.08rem; max-width: 68ch; margin: -.4rem 0 1.4rem; line-height:1.55; }

/* stat strip */
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1px;
         background: var(--line); border:1px solid var(--line); border-radius: 12px; overflow:hidden; margin: .4rem 0 1.6rem; }
.stat { background: var(--card); padding: 1rem 1.1rem; }
.stat .v { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.6rem; color:var(--ink); line-height:1.15; }
.stat .k { color: var(--muted); font-size: .9rem; margin-top:.15rem; }

/* takeaway under each EDA chart */
.takeaway { border-left: 3px solid var(--amber); background: #FFFBF1; padding: .75rem 1rem;
            border-radius: 0 8px 8px 0; color: var(--ink); line-height: 1.5; margin: .2rem 0 1.4rem; }

.qnum { display:inline-block; min-width: 2rem; height: 2rem; line-height: 2rem; text-align:center;
        border-radius: 50%; background: var(--night); color: var(--amber); font-weight:700;
        font-family:'Bricolage Grotesque',sans-serif; margin-right:.6rem; }
.qtitle { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.15rem; color:var(--ink); }

.pill-good, .pill-bad { display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .65rem;
                        border-radius:999px; font-weight:600; font-size:.88rem; }
.pill-good { background: var(--good-soft); color: var(--good); }
.pill-bad  { background: var(--caution-soft); color: var(--caution); }

.note { color: var(--muted); font-size: .92rem; }

/* bordered containers read as cards */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); }

[data-testid="stMetricValue"] { font-family:'Bricolage Grotesque',sans-serif; }

.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: var(--night); border: 1px solid var(--night); color: #fff; font-weight: 600;
  padding: .55rem 1.3rem;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  background: #262F57; border-color: #262F57; color: var(--amber);
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""

BRAND = """
<div class="brand">
  <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
    <rect x="3" y="12" width="9" height="19" rx="1.5" fill="#3A4577"/>
    <rect x="13" y="5" width="9" height="26" rx="1.5" fill="#4B5893"/>
    <rect x="23" y="15" width="8" height="16" rx="1.5" fill="#3A4577"/>
    <rect x="15.5" y="9" width="2" height="2" fill="#F2B544"/>
    <rect x="18.5" y="13" width="2" height="2" fill="#F2B544"/>
    <rect x="5.5" y="16" width="2" height="2" fill="#F2B544"/>
    <rect x="25.5" y="19" width="2" height="2" fill="#F2B544"/>
  </svg>
  <div><b>Real Estate<br>Investment Advisor</b><small>India housing · 5-year outlook</small></div>
</div>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, lede: str | None = None):
    st.markdown(f"# {title}")
    if lede:
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def stat_strip(items):
    """items = [(value, label), ...]"""
    cells = "".join(f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>'
                    for v, k in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def takeaway(text: str):
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)
