import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from plot_theme import SERIES, SEQUENTIAL, MUTED
from utils import style
from utils.data import load_models, load_mlflow_runs, models_ready, pretty

style.page_header("Model lab",
                  "Three models were tried for each job and tracked in MLflow. The best one by "
                  "F1 (verdict) and RMSE (price) is what the advisor uses.")

if not models_ready():
    st.error("No trained models yet. Run `python src/train.py`.")
    st.stop()

_, _, meta = load_models()
st.caption(f"Trained {meta['trained_at']} on {meta['rows'] - meta['test_rows']:,} listings, "
           f"tested on {meta['test_rows']:,} held-out listings.")

tab_c, tab_r, tab_ml = st.tabs(["Good Investment (classification)", "5-year price (regression)",
                                "MLflow runs"])

# ------------------------------------------------------------------ classification
with tab_c:
    res = meta["classification"]["results"]
    best = meta["classification"]["best"]
    b = next(r for r in res if r["model"] == best)
    style.stat_strip([
        (f"{b['accuracy']:.1%}", f"Accuracy ({best})"),
        (f"{b['precision']:.1%}", "Precision: when it says good, it's right"),
        (f"{b['recall']:.1%}", "Recall: share of good buys it finds"),
        (f"{b['f1']:.3f}", "F1 score"),
        (f"{b['roc_auc']:.3f}", "ROC AUC"),
    ])

    table = pd.DataFrame([{k: r[k] for k in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fit_seconds"]}
                          for r in res])
    table["model"] = table["model"].map(lambda m: f"{m}  ★" if m == best else m)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "model": "Model", "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
        "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
        "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
        "f1": st.column_config.NumberColumn("F1", format="%.4f"),
        "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
        "fit_seconds": st.column_config.NumberColumn("Train time (s)", format="%.0f")})

    a, c = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        for i, r in enumerate(res):
            fig.add_trace(go.Scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                                     name=f"{r['model']} (AUC {r['roc_auc']:.3f})",
                                     line=dict(color=SERIES[i], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                 line=dict(color=MUTED, dash="dot", width=1)))
        fig.update_layout(title="ROC curves", height=400, xaxis_title="False positive rate",
                          yaxis_title="True positive rate", legend=dict(y=-0.25))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c:
        cm = b["confusion_matrix"]
        fig = px.imshow(cm, text_auto=",", color_continuous_scale=SEQUENTIAL,
                        x=["Think twice", "Good"], y=["Think twice", "Good"],
                        labels=dict(x="Predicted", y="Actual", color="Listings"))
        fig.update_layout(title=f"Confusion matrix - {best}", height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    imp = pd.Series(b["importance"]).head(12)[::-1]
    fig = px.bar(x=imp.values, y=[pretty(i) for i in imp.index], orientation="h",
                 color_discrete_sequence=[SERIES[0]], labels={"x": "Share of importance", "y": ""})
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(title=f"What {best} relies on most", height=420)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    style.takeaway(f"{pretty(imp.index[-1])} and {pretty(imp.index[-2]).lower()} carry the most "
                   f"weight - which makes sense, because the Good Investment label is built from "
                   f"expected growth and how fairly a home is priced against its city.")

# ------------------------------------------------------------------ regression
with tab_r:
    res = meta["regression"]["results"]
    best = meta["regression"]["best"]
    b = next(r for r in res if r["model"] == best)
    style.stat_strip([
        (f"₹{b['rmse']:.2f} L", f"RMSE ({best})"),
        (f"₹{b['mae']:.2f} L", "Mean absolute error"),
        (f"{b['mape']:.1%}", "Mean absolute % error"),
        (f"{b['r2']:.4f}", "R²"),
    ])
    table = pd.DataFrame([{k: r[k] for k in ["model", "rmse", "mae", "mape", "r2", "fit_seconds"]} for r in res])
    table["model"] = table["model"].map(lambda m: f"{m}  ★" if m == best else m)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "model": "Model", "rmse": st.column_config.NumberColumn("RMSE (₹ L)", format="%.3f"),
        "mae": st.column_config.NumberColumn("MAE (₹ L)", format="%.3f"),
        "mape": st.column_config.NumberColumn("MAPE", format="percent"),
        "r2": st.column_config.NumberColumn("R²", format="%.4f"),
        "fit_seconds": st.column_config.NumberColumn("Train time (s)", format="%.0f")})

    a, c = st.columns(2, gap="large")
    with a:
        sc = b["scatter"]
        mx = max(max(sc["actual"]), max(sc["predicted"]))
        fig = go.Figure(go.Scattergl(x=sc["actual"], y=sc["predicted"], mode="markers",
                                     marker=dict(size=5, color=SERIES[0], opacity=.45),
                                     hovertemplate="actual ₹%{x:.1f} L<br>predicted ₹%{y:.1f} L<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=SERIES[1], width=2),
                                 hoverinfo="skip"))
        fig.update_layout(title="Actual vs predicted (held-out listings)", height=420, showlegend=False,
                          xaxis_title="Actual 5-year price (₹ L)", yaxis_title="Predicted (₹ L)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c:
        imp = pd.Series(b["importance"]).head(10)[::-1]
        fig = px.bar(x=imp.values, y=[pretty(i) for i in imp.index], orientation="h",
                     color_discrete_sequence=[SERIES[0]], labels={"x": "Share of importance", "y": ""})
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(title=f"What {best} relies on most", height=420)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    style.takeaway("Today's price explains most of the future price, as you'd expect. What the "
                   "model adds is the growth on top: the features behind expected appreciation "
                   "decide whether a ₹60 L flat becomes ₹80 L or ₹95 L.")

# ------------------------------------------------------------------ mlflow
with tab_ml:
    runs = load_mlflow_runs()
    if runs is None:
        st.info("No MLflow export found yet - run `python src/train.py`.")
    else:
        runs = runs.rename(columns=lambda c: c.replace("metrics.", "").replace("tags.", ""))
        st.dataframe(runs, hide_index=True, width="stretch")
    reg_c, reg_r = meta["classification"].get("registry_version"), meta["regression"].get("registry_version")
    st.markdown(
        "Every run logs its parameters, metrics, confusion matrix or actual-vs-predicted plot, "
        "feature importance and the fitted pipeline. The winners are registered as "
        f"**RealEstate_GoodInvestment_Classifier** (v{reg_c}) and "
        f"**RealEstate_FuturePrice_Regressor** (v{reg_r}) with the alias `champion`.")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db\n# then open http://127.0.0.1:5000",
            language="bash")
