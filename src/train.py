"""
Step 3 + 4 - model training, evaluation and MLflow tracking.

    python src/train.py            # full run
    python src/train.py --quick    # 60k-row sample, handy while tinkering

Two jobs:
  * Classification -> is this a Good Investment? (Logistic Regression,
    Random Forest, XGBoost)
  * Regression     -> what will it be worth in 5 years? (Linear Regression,
    Random Forest, XGBoost)

Every model is logged to MLflow (params, metrics, plots, the model itself).
The best of each is registered in the MLflow Model Registry and also saved
to models/ as a joblib file, which is what the Streamlit app loads.

To browse the runs afterwards:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import tempfile
import time
import warnings

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from config import (CLEAN_PARQUET, MLFLOW_EXPERIMENT, MLFLOW_URI, MODELS_DIR, RANDOM_STATE,
                    REPORTS_DIR, GOOD_GROWTH_THRESHOLD, MAX_PRICE_VS_CITY, FORECAST_YEARS,
                    AMENITY_LIST)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# Locality is left out on purpose: 500 levels with a few hundred rows each
# would mostly teach the model noise. City carries the location signal.
CAT_FEATURES = ["State", "City", "Property_Type", "Furnished_Status",
                "Public_Transport_Accessibility", "Facing", "Owner_Type", "Availability_Status"]
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
                "Floor_No", "Total_Floors", "Floor_Ratio", "Size_per_BHK",
                "Nearby_Schools", "Nearby_Hospitals", "Transport_Score",
                "School_Density_Score", "Infrastructure_Score",
                "Parking_Space", "Security", "Amenity_Count",
                *[f"Has_{a}" for a in AMENITY_LIST],
                "Is_Ready_To_Move", "Price_vs_City"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])


def candidates(task):
    if task == "clf":
        return {
            "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
            "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=5,
                                                   max_samples=0.5, n_jobs=-1, random_state=RANDOM_STATE),
            "XGBoost": XGBClassifier(n_estimators=450, max_depth=6, learning_rate=0.08, subsample=0.9,
                                     colsample_bytree=0.9, tree_method="hist", eval_metric="logloss",
                                     n_jobs=-1, random_state=RANDOM_STATE),
        }
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=3,
                                              max_samples=0.5, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=700, max_depth=7, learning_rate=0.06, subsample=0.9,
                                colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
                                random_state=RANDOM_STATE),
    }


def grouped_importance(pipe) -> pd.Series:
    """Roll one-hot columns back up to the original feature (City_Mumbai -> City)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        coef = model.coef_
        vals = np.abs(coef[0] if coef.ndim > 1 else coef)
    base = []
    for n in names:
        kind, col = n.split("__", 1)
        if kind == "cat":
            col = next(c for c in CAT_FEATURES if col.startswith(c + "_") or col == c)
        base.append(col)
    imp = pd.Series(vals, index=base).groupby(level=0).sum()
    return (imp / imp.sum()).sort_values(ascending=False)


def _save_fig(fig, folder, name):
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def run_task(task, X_tr, X_te, y_tr, y_te, sample_input):
    target = "Good_Investment" if task == "clf" else "Future_Price_5Y"
    results, fitted = [], {}

    for name, model in candidates(task).items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=f"{'clf' if task == 'clf' else 'reg'}_{name}") as run:
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            fit_secs = time.time() - t0

            mlflow.set_tags({"task": "classification" if task == "clf" else "regression",
                             "target": target, "model_family": name})
            mlflow.log_params({k: v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool)) and v is not None})
            mlflow.log_param("n_train_rows", len(X_tr))

            with tempfile.TemporaryDirectory() as tmp:
                if task == "clf":
                    proba = pipe.predict_proba(X_te)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    m = {"accuracy": accuracy_score(y_te, pred),
                         "precision": precision_score(y_te, pred),
                         "recall": recall_score(y_te, pred),
                         "f1": f1_score(y_te, pred),
                         "roc_auc": roc_auc_score(y_te, proba)}
                    cm = confusion_matrix(y_te, pred)
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    ax.imshow(cm, cmap="Blues")
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{v:,}", ha="center", va="center",
                                color="white" if v > cm.max() / 2 else "black")
                    ax.set_xticks([0, 1], ["Not good", "Good"])
                    ax.set_yticks([0, 1], ["Not good", "Good"])
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
                    mlflow.log_artifact(_save_fig(fig, tmp, "confusion_matrix.png"))
                    extra = {"confusion_matrix": cm.tolist()}
                    fpr, tpr, _ = roc_curve(y_te, proba)
                    keep = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
                    extra["roc"] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}
                else:
                    pred = pipe.predict(X_te)
                    m = {"rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                         "mae": mean_absolute_error(y_te, pred),
                         "r2": r2_score(y_te, pred),
                         "mape": mean_absolute_percentage_error(y_te, pred)}
                    fig, ax = plt.subplots(figsize=(4.5, 4))
                    idx = np.random.default_rng(0).choice(len(pred), min(3000, len(pred)), replace=False)
                    ax.scatter(np.asarray(y_te)[idx], pred[idx], s=4, alpha=.4)
                    lim = [0, max(np.max(y_te), np.max(pred))]
                    ax.plot(lim, lim, color="#eb6834")
                    ax.set_xlabel("Actual (₹ lakhs)"); ax.set_ylabel("Predicted (₹ lakhs)"); ax.set_title(name)
                    mlflow.log_artifact(_save_fig(fig, tmp, "actual_vs_predicted.png"))
                    ratio = np.asarray(y_te) / pred
                    extra = {"interval": {"low": float(np.quantile(ratio, 0.05)),
                                          "high": float(np.quantile(ratio, 0.95))},
                             "scatter": {"actual": np.asarray(y_te)[idx][:1500].round(2).tolist(),
                                         "predicted": pred[idx][:1500].round(2).tolist()}}

                m["fit_seconds"] = fit_secs
                mlflow.log_metrics(m)

                imp = grouped_importance(pipe)
                imp_path = os.path.join(tmp, "feature_importance.csv")
                imp.rename("importance").to_csv(imp_path)
                mlflow.log_artifact(imp_path)

            signature = infer_signature(sample_input, pipe.predict(sample_input))
            # cloudpickle: these are my own models, and skops refuses tree models
            # unless every internal type is whitelisted
            mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                                     input_example=sample_input.head(3),
                                     serialization_format="cloudpickle")

        shown = ", ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "fit_seconds")
        print(f"  {name:<20} {shown}  ({fit_secs:.0f}s)")
        results.append({"model": name, "run_id": run.info.run_id, **m,
                        "importance": imp.round(4).to_dict(), **extra})
        fitted[name] = pipe

    key = "f1" if task == "clf" else "rmse"
    best = (max if task == "clf" else min)(results, key=lambda r: r[key])
    return best, results, fitted[best["model"]]


def register(run_id, model_name):
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        mlflow.MlflowClient().set_registered_model_alias(model_name, "champion", mv.version)
        return int(mv.version)
    except Exception as err:   # registry is a nice-to-have, never block the pipeline on it
        print(f"  (model registry skipped: {err})")
        return None


def main(quick=False):
    df = pd.read_parquet(CLEAN_PARQUET)
    if quick:
        df = df.sample(60_000, random_state=RANDOM_STATE)
    print(f"Training on {len(df):,} rows, {len(FEATURES)} input features")

    X = df[FEATURES]
    y_clf = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    sample = X_te.head(50)

    print("\nClassification - Good Investment")
    best_c, res_c, pipe_c = run_task("clf", X_tr, X_te, yc_tr, yc_te, sample)
    print("\nRegression - price in 5 years")
    best_r, res_r, pipe_r = run_task("reg", X_tr, X_te, yr_tr, yr_te, sample)

    print(f"\nBest classifier: {best_c['model']} (F1 {best_c['f1']:.4f})")
    print(f"Best regressor : {best_r['model']} (RMSE {best_r['rmse']:.3f} lakhs)")
    v_c = register(best_c["run_id"], "RealEstate_GoodInvestment_Classifier")
    v_r = register(best_r["run_id"], "RealEstate_FuturePrice_Regressor")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe_c, MODELS_DIR / "classifier.joblib", compress=3)
    joblib.dump(pipe_r, MODELS_DIR / "regressor.joblib", compress=3)

    # everything the app needs to describe the models, without touching MLflow
    form_options = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_FEATURES}
    form_options["state_city"] = df.groupby("State")["City"].unique().apply(sorted).apply(list).to_dict()
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "rows": len(df), "test_rows": len(X_te),
        "features": FEATURES, "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
        "rules": {"growth_threshold": GOOD_GROWTH_THRESHOLD, "max_price_vs_city": MAX_PRICE_VS_CITY,
                  "years": FORECAST_YEARS},
        "classification": {"best": best_c["model"], "registry_version": v_c, "results": res_c},
        "regression": {"best": best_r["model"], "registry_version": v_r, "results": res_r},
        "form_options": form_options,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=1, default=float))

    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
    keep = [c for c in runs.columns if c.startswith(("metrics.", "tags.task", "tags.model_family"))]
    runs[["run_id", "start_time", *keep]].to_csv(REPORTS_DIR / "mlflow_runs.csv", index=False)
    print(f"\nSaved models -> {MODELS_DIR}\nMLflow runs  -> {MLFLOW_URI}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="train on a 60k sample")
    main(**vars(ap.parse_args()))
