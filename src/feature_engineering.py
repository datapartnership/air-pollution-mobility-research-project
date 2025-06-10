import re
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# regex to pull YYYY-MM-DD
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
    # drop any empty/all-NA columns, then concat
    cleaned = [df.dropna(axis=1, how="all") for df in records]
    return pd.concat(cleaned, ignore_index=True)

def make_lag_features(df, id_col="geom_id", target="no2_mean", nlags=7):
    df = df.sort_values([id_col, "date"])
    for lag in range(1, nlags+1):
        df[f"{target}_lag{lag}"] = df.groupby(id_col)[target].shift(lag)
    return df

class NeighborAggregator:
    """
    Computes k-nearest neighbors in EPSG:3857 and
    for each row aggregates all lag-columns over its neighbors.
    """
    def __init__(self, k=8, id_col="geom_id"):
        self.k = k
        self.id_col = id_col

    def fit(self, X: gpd.GeoDataFrame, y=None):
        # Project to metric CRS to get correct distances
        proj = X.to_crs("EPSG:3857")
        centroids = proj.geometry.centroid.apply(lambda p: (p.x, p.y)).tolist()
        self.ids_ = proj[self.id_col].values
        # use named argument here
        self.nn = NearestNeighbors(n_neighbors=self.k + 1)
        self.nn.fit(centroids)
        # each row’s neighbors, skip the first (itself)
        self.neigh_idx = self.nn.kneighbors(centroids, return_distance=False)[:, 1:]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # identify lag columns
        lag_cols = [c for c in X.columns if c.startswith("no2_mean_lag")]
        rows = []
        for neigh in self.neigh_idx:
            # positional indexing of neighbor rows
            df_neigh = X.iloc[neigh]
            rows.append(df_neigh[lag_cols].mean().to_dict())
        return pd.DataFrame(rows, index=X.index).add_prefix("neigh_")
