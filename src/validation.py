from pathlib import Path
import geopandas as gpd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Path configuration (aligns with notebook variables)
# -----------------------------------------------------------------------------
CURR_PATH = Path(__file__).resolve()
REPO_PATH = CURR_PATH.parent.parent
DATA_PATH = REPO_PATH / "data"

# Server-side folders for OSM, ESA and outputs
CITY_CONFIG = {
    "addis-ababa": {
        "name": "addis-ababa",
        "osm_folder": DATA_PATH / "Populated meshes" / "addis-mesh-data",
        "esa_folder": DATA_PATH / "addis-ababa-ESA-landcover",
        "out_folder": DATA_PATH / "addis-ababa-luis"
    },
    "baghdad": {
        "name": "baghdad",
        "osm_folder": DATA_PATH / "Populated meshes" / "baghdad-mesh-data",
        "esa_folder": DATA_PATH / "baghdad-ESA-landcover",
        "out_folder": DATA_PATH / "baghdad-luis"
    }
}

BUILT_UP_COL = "built_up_a"
ESA_CATEGORIES = [
  "tree_cover_a", "shrubland_a", "grassland_a", "sparse_veg_a",
  "snow_a", "water_bod_a", "wetland_a", "mangroves_a",
  "moss_a", "unclassified_a",
]
OSM_BLD_COLS = [
    "lu_commercial_area", "lu_industrial_area", "lu_residential_area"
]
ROAD_COLS = ["road_len", "road_share"]
POI_COLS = ["poi_count", "poi_share"]

# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------
def merge_meshes(osm_fp: Path, esa_fp: Path) -> gpd.GeoDataFrame:
    """
    Merge OSM and ESA meshes on 'geom_id', compute validation metrics
    and derive 'non_built_area'.
    """
    # Load datasets
    osm_gdf = gpd.read_file(osm_fp)
    esa_gdf = gpd.read_file(esa_fp)

    # Align CRS
    if osm_gdf.crs != esa_gdf.crs:
        esa_gdf = esa_gdf.to_crs(osm_gdf.crs)

    # Drop ESA geometry to keep primary geometry from OSM
    esa_gdf = esa_gdf.drop(columns=[esa_gdf.geometry.name])

    # Join on geom_id
    merged = osm_gdf.merge(esa_gdf, on="geom_id", how="inner")

    # Handle nulls and negative sentinel values
    merged[OSM_BLD_COLS] = merged[OSM_BLD_COLS].clip(lower=0).fillna(0)
    merged[BUILT_UP_COL] = (
        merged[BUILT_UP_COL].clip(lower=0)
        .fillna(0)
    )

    # Compute OSM aggregated built area
    merged["osm_built_sum"] = merged[OSM_BLD_COLS].sum(axis=1)
    # ESA ground-truth built area
    merged["built_up_esa"] = merged[BUILT_UP_COL]

    # Validation errors
    merged["abs_err"] = (merged["osm_built_sum"] - merged["built_up_esa"]).abs()
    merged["rel_err"] = (
        merged["abs_err"]
        / merged["built_up_esa"].replace({0: np.nan})
    )

    # Non-built area from ESA
    merged["non_built_area"] = merged[ESA_CATEGORIES].sum(axis=1)

    return merged


def prune_and_save(merged: gpd.GeoDataFrame, out_fp: Path) -> None:
    """
    Retain only road, POI, land-use and non-built columns + geometry,
    then write to GeoPackage (overwrite mode).
    """
    # Define schema for output
    save_cols = [
        "geom_id",
        *ROAD_COLS,
        *POI_COLS,
        *OSM_BLD_COLS,
        "non_built_area",
        merged.geometry.name
    ]
    pruned = merged[save_cols]

    # Ensure directory hierarchy exists
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    # Write, overwriting if exists
    pruned.to_file(out_fp, driver="GPKG")


def process_date(osm_fp: Path, config: dict) -> None:
    """
    Process a single date-file: merge, validate, print summary and save output.
    """
    city = config["name"]
    # Derive matching ESA and output paths
    date_tag = osm_fp.stem.removeprefix(f"{city}-")
    esa_fp = config["esa_folder"] / osm_fp.name
    out_fp = config["out_folder"] / osm_fp.name

    # Merge and validate
    merged = merge_meshes(osm_fp, esa_fp)

    # Compute summary statistics
    avg_abs = merged["abs_err"].mean()
    avg_rel = merged["rel_err"].mean()
    print(
        f"[{city} | {date_tag}] "
        f"Avg abs error: {avg_abs:.2f} m², "
        f"Avg rel error: {avg_rel:.2%}"
    )

    # Prune attributes and persist
    prune_and_save(merged, out_fp)


def process_city(city: str, max_workers: int = 4) -> None:
    """
    Execute parallel validation for all date-specific GeoPackages of a city.
    """
    config = CITY_CONFIG[city]
    # Ensure output directory
    config["out_folder"].mkdir(parents=True, exist_ok=True)

    # Gather OSM files
    osm_files = sorted(
        config["osm_folder"].glob(f"{city}-*.gpkg")
    )

    # Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_date, osm_fp, config)
            for osm_fp in osm_files
        ]
        # Ensure all tasks complete (raises if any fail)
        for fut in as_completed(futures):
            fut.result()

# -----------------------------------------------------------------------------
# Convenience entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Orchestrate for both cities
    for city_key in CITY_CONFIG:
        process_city(city_key)
