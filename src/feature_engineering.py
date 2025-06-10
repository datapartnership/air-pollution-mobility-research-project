import re
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shap

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

def load_mesh_series(folder: Path,
                     id_col: str = "geom_id",
                     target: str = "no2_mean",
                     features: list[str] | None = None) -> pd.DataFrame:
    """
    Load all .gpkg files in `folder`, extract date from filename,
    keep id_col, target, geometry + any `features` you pass,
    then drop all *_share columns and `fid`.
    """
    records = []
    for fp in sorted(folder.glob("*.gpkg")):
        m = DATE_RE.search(fp.stem)
        if not m:
            raise ValueError(f"No YYYY-MM-DD in filename {fp.name}")
        date = pd.to_datetime(m.group(0))
        gdf = gpd.read_file(fp)

        # select only id, target, geometry, date, plus any extra features
        keep = {id_col, target, "geometry"}
        if features:
            keep |= set(features)
        # ensure date is added
        gdf = gdf.loc[:, [c for c in gdf.columns if c in keep]].copy()
        gdf["date"] = date
        records.append(gdf)

    df = pd.concat(records, ignore_index=True)

    # drop any *_share fields and fid
    drop_cols = [c for c in df.columns if c.endswith("_share") or c == "fid"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # drop columns that were all‐NA (e.g. if feature missing)
    df = df.dropna(axis=1, how="all")
    return df

def make_lag_features(df, id_col="geom_id", target="no2_mean", nlags=7):
    df = df.sort_values([id_col, "date"])
    for lag in range(1, nlags+1):
        df[f"{target}_lag{lag}"] = df.groupby(id_col)[target].shift(lag)
    return df

class NeighborAggregator:
    """
    Vectorised neighbour-lag aggregator: for each (geom_id,date) row
    computes the mean of all its lag-features among its k nearest
    neighbours _on that same date_ in one bulk merge (no python loop).
    """
    def __init__(self, k=8, id_col="geom_id"):
        self.k = k
        self.id_col = id_col

    def fit(self, static_gdf: gpd.GeoDataFrame, y=None):
        # project for metric distances
        m = static_gdf.to_crs("EPSG:3857")
        pts = np.vstack(m.geometry.centroid.apply(lambda p: (p.x, p.y)).values)
        self.ids_ = m[self.id_col].values
        self.nn = NearestNeighbors(n_neighbors=self.k+1).fit(pts)
        # each row’s neighbors (skip self at index 0)
        self.neigh_idx = self.nn.kneighbors(pts, return_distance=False)[:, 1:]
        return self

    def transform(self, df: pd.DataFrame, target="no2_mean") -> pd.DataFrame:
        # identify lag cols
        lag_cols = sorted([c for c in df if c.startswith(f"{target}_lag")])
        # build a flat neighbor‐map table
        src = np.repeat(self.ids_, self.k)
        dst = self.ids_[self.neigh_idx.flatten()]
        neigh_map = pd.DataFrame({self.id_col: src, "neigh_id": dst})

        # original lag‐features + date
        core = df[[self.id_col, "date"] + lag_cols]
        # neighbour lag‐features by merge
        merged = (
            neigh_map
            .merge(core, on=self.id_col)
            .rename(columns={self.id_col: "orig_id"})
            .merge(core.rename(columns={self.id_col: "neigh_id"}), 
                   on=["neigh_id","date"], 
                   suffixes=("","_nbr"))
        )
        # average per original id+date
        agg = (
            merged
            .groupby(["orig_id","date"])
            [[c+"_nbr" for c in lag_cols]]
            .mean()
            .rename(columns=lambda c: "neigh_"+c.replace("_nbr",""))
            .reset_index()
            .rename(columns={"orig_id": self.id_col})
        )
        # merge back
        return df.merge(agg, on=[self.id_col,"date"], how="left")
    
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── 1) TRAIN RF PIPELINE ─────────────────────────────────────────────────────
def train_rf_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_params: dict | None = None
) -> Pipeline:
    rf_params = rf_params or {
        "n_estimators": 200,
        "max_depth": 15,
        "n_jobs": -1,
        "random_state": 42,
        "oob_score": True
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(**rf_params))
    ])
    pipe.fit(X_train, y_train)
    return pipe

# ─── 2) EVALUATE METRICS ──────────────────────────────────────────────────────
def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    rf = pipeline.named_steps["rf"]
    y_pred = pipeline.predict(X_test)
    return {
        "oob_r2": getattr(rf, "oob_score_", None),
        "test_r2": r2_score(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }

# ─── 3) SHAP EXPLANATION & SUMMARY PLOTS ──────────────────────────────────────
def explain_shap(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    max_display: int = 15
) -> shap.Explanation:
    """
    Returns a SHAP Explanation object on X_eval (using X_background for background data),
    and emits the bar + beeswarm summary plots.
    """
    explainer = shap.TreeExplainer(
        pipeline.named_steps["rf"],
        data=X_background,
        feature_perturbation="interventional"
    )
    shap_exp = explainer(X_eval)  # returns an Explanation object
    shap.plots.bar(shap_exp, max_display=max_display)
    shap.plots.beeswarm(shap_exp, max_display=max_display)
    return shap_exp

# ─── 4) DEPENDENCE PLOTS FOR TOP-K FEATURES ────────────────────────────────────
def plot_shap_dependence(
    shap_exp: shap.Explanation,
    X_eval: pd.DataFrame,
    top_k: int = 2
) -> None:
    """
    Computes the top_k features by mean(|SHAP|) and renders dependence_plot for each.
    """
    # 1) compute mean absolute SHAP per feature
    mean_abs = np.abs(shap_exp.values).mean(axis=0)
    # 2) identify top_k feature names
    top_feats = X_eval.columns[np.argsort(mean_abs)[-top_k:]]
    # 3) loop and plot
    for feat in top_feats:
        shap.dependence_plot(feat, shap_exp, X_eval)

# ─── 5) ELASTICITIES VIA SHAP ─────────────────────────────────────────────────
def compute_elasticities_shap(
    pipeline: Pipeline,
    shap_exp: shap.Explanation,
    low_q: float = 0.25,
    high_q: float = 0.75
) -> pd.DataFrame:
    """
    Approximate elasticities eᵢ = medianₘ [(ϕᵢₘ / ŷₘ) * (xᵢₘ / Δxᵢ)]  
    where Δxᵢ = Q₀.₇₅(xᵢ) – Q₀.₂₅(xᵢ).
    """
    # unpack
    X = pd.DataFrame(shap_exp.data, columns=shap_exp.feature_names)
    shap_vals = shap_exp.values
    # predictions
    y_hat = pipeline.predict(X)
    # interquartile
    qs = X.quantile([low_q, high_q])
    dx = qs.loc[high_q] - qs.loc[low_q]
    # compute elementwise elasticity per sample/feature
    rels = (shap_vals / y_hat[:, None]) * (X.values / dx.values[None, :])
    elas = pd.DataFrame({
        "feature": X.columns,
        "median_elasticity": np.median(rels, axis=0),
        "p10": np.percentile(rels, 10, axis=0),
        "p90": np.percentile(rels, 90, axis=0),
    })
    return elas.sort_values("median_elasticity", ascending=False)
