# ─── in src/feature_engineering.py ────────────────────────────────────────

import re
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

def load_mesh_series(folder: Path, id_col="geom_id", target="no2_mean"):
    records = []
    for fp in sorted(folder.glob("*.gpkg")):
        m = DATE_RE.search(fp.stem)
        if not m:
            raise ValueError(f"No YYYY-MM-DD in filename {fp.name}")
        date_str = m.group(0)
        gdf = gpd.read_file(fp)[[id_col, target, "geometry"]].copy()
        gdf["date"] = pd.to_datetime(date_str)
        records.append(gdf)
    cleaned = [df.dropna(axis=1, how="all") for df in records]
    return pd.concat(cleaned, ignore_index=True)

def make_lag_features(df, id_col="geom_id", target="no2_mean", nlags=7):
    df = df.sort_values([id_col, "date"])
    for lag in range(1, nlags+1):
        df[f"{target}_lag{lag}"] = df.groupby(id_col)[target].shift(lag)
    return df

class NeighborAggregator:
    """
    Computes k-nearest neighbors in EPSG:3857 and for each row
    averages its lag-features among its neighbors on the same date.
    """
    def __init__(self, k=8, id_col="geom_id"):
        self.k = k
        self.id_col = id_col

    def fit(self, static_gdf: gpd.GeoDataFrame, y=None):
        # Project to metric CRS for accurate distances
        m = static_gdf.to_crs("EPSG:3857")
        coords = list(m.geometry.centroid.apply(lambda p: (p.x, p.y)))
        self.ids_ = m[self.id_col].values
        # Build an id → position lookup
        self.id_to_pos = {cid: i for i, cid in enumerate(self.ids_)}
        # Fit neighbors
        self.nn = NearestNeighbors(n_neighbors=self.k + 1)
        self.nn.fit(coords)
        self.neigh_idx = self.nn.kneighbors(coords, return_distance=False)[:, 1:]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_cols = [c for c in df.columns if c.startswith("no2_mean_lag")]
        out = np.full((len(df), len(lag_cols)), np.nan)

        # Loop row-by-row
        for i, (geom, date) in enumerate(zip(df[self.id_col], df["date"])):
            pos = self.id_to_pos[geom]
            neigh_pos = self.neigh_idx[pos]
            neigh_ids = self.ids_[neigh_pos]
            mask = (df[self.id_col].isin(neigh_ids)) & (df["date"] == date)
            neigh_block = df.loc[mask, lag_cols]
            if not neigh_block.empty:
                out[i, :] = neigh_block.mean().values

        cols = [f"neigh_{c}" for c in lag_cols]
        return pd.DataFrame(out, index=df.index, columns=cols)
