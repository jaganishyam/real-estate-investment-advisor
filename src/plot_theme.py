"""
One look for every chart - used by the EDA export and the Streamlit app.
"""

import plotly.graph_objects as go
import plotly.io as pio

INK = "#1A1F36"
MUTED = "#5B6382"
GRID = "#E6E8F0"
SURFACE = "#FFFFFF"

# categorical order matters (it's what keeps neighbours colour-blind safe),
# so charts take colours from the front of this list, never shuffled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

GOOD = "#1F8A70"      # good investment
CAUTION = "#C2413B"   # think twice
AMBER = "#F2B544"     # the lit-window colour from the skyline

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#C2413B"], [0.25, "#E7A39C"], [0.5, "#F1F1F3"], [0.75, "#86b6ef"], [1, "#1c5cab"]]

FONT = "Source Sans 3, Source Sans Pro, Segoe UI, Helvetica, Arial, DejaVu Sans, sans-serif"

pio.templates["advisor"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left"),
        xaxis=dict(automargin=True, gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                   title=dict(font=dict(color=MUTED))),
        yaxis=dict(automargin=True, gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                   title=dict(font=dict(color=MUTED))),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, family=FONT)),
        margin=dict(l=70, r=24, t=64, b=56),
        bargap=0.25,
        barcornerradius=4,
    )
)
pio.templates.default = "advisor"


def money(lakhs: float) -> str:
    """45.2 -> '₹45.2 L', 152 -> '₹1.52 Cr'."""
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Cr"
    return f"₹{lakhs:.1f} L"


def rupees(x: float) -> str:
    return f"₹{x:,.0f}"
