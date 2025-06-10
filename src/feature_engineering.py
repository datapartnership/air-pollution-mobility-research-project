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
    
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── New functions ──────────────────────────────────────────────────────────

def train_rf_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    rf_params: dict | None = None
) -> Pipeline:
    """
    Build and fit a standard scaler + RF pipeline.
    Returns the fitted Pipeline.
    """
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

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Compute OOB (if available), test R² and RMSE.
    """
    rf = pipeline.named_steps["rf"]
    preds = pipeline.predict(X_test)
    return {
        "oob_r2": getattr(rf, "oob_score_", None),
        "test_r2": r2_score(y_test, preds),
        "test_rmse": np.sqrt(mean_squared_error(y_test, preds))
    }

def explain_shap(
    pipeline: Pipeline,
    X_bg: pd.DataFrame,
    X_eval: pd.DataFrame,
    max_display: int = 15
) -> tuple[shap.Explainer, np.ndarray]:
    """
    Compute and plot SHAP summary (bar + beeswarm).
    Returns (explainer, shap_values).
    """
    explainer = shap.TreeExplainer(
        pipeline.named_steps["rf"],
        data=X_bg,
        feature_perturbation="interventional"
    )
    shap_vals = explainer.shap_values(X_eval)
    # summary visuals
    shap.plots.bar(shap_vals, max_display=max_display)
    shap.plots.beeswarm(shap_vals, max_display=max_display)
    return explainer, shap_vals

def plot_shap_dependence(
    explainer: shap.Explainer,
    shap_vals: np.ndarray,
    X_eval: pd.DataFrame,
    top_k: int = 2
) -> None:
    """
    Identify top_k features by mean(|SHAP|) and render dependence plots.
    """
    mean_abs = np.abs(shap_vals).mean(0)
    top_feats = np.array(X_eval.columns)[np.argsort(mean_abs)[-top_k:]]
    for feat in top_feats:
        shap.dependence_plot(feat, shap_vals, X_eval)

def compute_elasticities_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    shap_vals: np.ndarray,
    low_q: float = 0.25,
    high_q: float = 0.75
) -> pd.DataFrame:
    """
    Approximate elasticities via SHAP:
      eᵢ = median over samples of [ (ϕᵢ / ŷ) * (xᵢ / Δxᵢ) ]
    where Δxᵢ = Q_hi - Q_lo.
    """
    y_hat = pipeline.predict(X)
    qs = X.quantile([low_q, high_q])
    dx = qs.loc[high_q] - qs.loc[low_q]

    rels = (shap_vals / y_hat[:, None]) * (X.values / dx.values[None, :])
    elas = pd.DataFrame({
        "feature": X.columns,
        "median_elasticity": np.median(rels, axis=0),
        "p10":             np.percentile(rels, 10, axis=0),
        "p90":             np.percentile(rels, 90, axis=0),
    })
    return elas.sort_values("median_elasticity", ascending=False)