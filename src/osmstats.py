"""
OSM → Mesh enrichment utility
─────────────────────────────
This version handles both raw OSM fields (e.g., highway, amenity, shop, landuse)
and Geofabrik “*-latest-free.shp” extracts where many tags collapse into 'fclass'.
Each function is documented inline to explain its purpose and logic.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Helper: pick the first existing column from a list of candidates
# ─────────────────────────────────────────────────────────────────────────────
def _pick_column(gdf: gpd.GeoDataFrame, candidates: List[str]) -> str:
    """
    Return the first column name from `candidates` that exists in `gdf`.
    If none of the candidates are present, raise a ValueError.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame to inspect.
    candidates : list[str]
        Column names to search for, in priority order.

    Returns
    -------
    str
        The first matching column name.

    Raises
    ------
    ValueError
        If none of the candidates exist in the GeoDataFrame.
    """
    for col in candidates:
        if col in gdf.columns:
            return col
    # If we reach here, none of the candidates were found
    raise ValueError(
        f"None of the columns {candidates} found in the supplied OSM layer. "
        f"Available columns: {list(gdf.columns)[:20]}…"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Load OSM layers into Roads, POIs, and Land-use GeoDataFrames
# ─────────────────────────────────────────────────────────────────────────────
def load_osm_layers(osm_shapefile: Path) -> dict:
    """
    Read a Geofabrik theme-shapefile (or any OSM-derived shapefile)
    and split it into three GeoDataFrames:
      • roads   – LineString / MultiLineString where highway/fclass ∈ major classes
      • pois    – Point geometries with amenity/shop/fclass matching at least some valid value
      • landuse – Polygon geometries whose `landuse` or `fclass` column is not null

    If a raw OSM shapefile with 'highway', 'amenity', 'shop', 'landuse' columns is passed,
    those will be used. If the shapefile only has 'fclass' (common for Geofabrik extracts),
    that single column will be used to identify roads, POIs, and land-use categories.

    Parameters
    ----------
    osm_shapefile : Path
        Path to the OSM-derived shapefile (e.g., 'ethiopia-latest-free.shp').

    Returns
    -------
    dict
        A dictionary containing three GeoDataFrames:
          - "roads":    Road segments (LineString/MultiLineString)
          - "pois":     Point-of-interest points (Point)
          - "landuse":  Land-use polygons (Polygon/MultiPolygon)
    """
    # Read the full shapefile into a GeoDataFrame
    gdf = gpd.read_file(str(osm_shapefile))

    # Ensure the coordinate reference system is EPSG:4326
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Identify which column holds road categories: either 'highway' or fallback to 'fclass'
    road_attr = _pick_column(gdf, ["highway", "fclass"])
    # Identify if 'amenity' exists; else, we may rely on 'shop' or 'fclass'
    amenity_col = "amenity" if "amenity" in gdf.columns else None
    shop_col = "shop" if "shop" in gdf.columns else None
    # Identify land-use column: either 'landuse' or fallback to 'fclass'
    land_col = _pick_column(gdf, ["landuse", "fclass"])

    # ――― ① Extract Roads ―――
    # Define major highway classes that we care about
    road_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
    # Filter rows where geometry is a line and the attribute matches one of our road_classes
    gdf_roads = gdf[
        (gdf.geometry.type.isin(["LineString", "MultiLineString"])) &
        (gdf[road_attr].isin(road_classes))
    ].copy()

    # ――― ② Extract POIs ―――
    # Start with all points
    poi_mask = gdf.geometry.type.eq("Point")
    if amenity_col:
        # If 'amenity' exists, keep points having any non-null amenity
        poi_mask &= gdf[amenity_col].notna()
    elif shop_col:
        # Else if 'shop' exists, keep points having any non-null shop
        poi_mask &= gdf[shop_col].notna()
    else:
        # Otherwise, fall back to any point that has a non-null fclass
        poi_mask &= gdf[road_attr].notna()

    gdf_pois = gdf[poi_mask].copy()

    # ――― ③ Extract Land-use Polygons ―――
    # Keep only Polygon/MultiPolygon geometries with a non-null landuse
    landuse_mask = (
        gdf.geometry.type.isin(["Polygon", "MultiPolygon"]) &
        gdf[land_col].notna()
    )
    gdf_land = gdf[landuse_mask].copy()

    # ――― ④ Harmonise Column Names ―――
    # Rename whichever column we picked for roads to 'fclass' so downstream code is uniform
    gdf_roads.rename(columns={road_attr: "fclass"}, inplace=True)

    # For POIs, ensure we have both 'amenity' and 'shop' columns (fill with None if missing)
    if amenity_col:
        gdf_pois.rename(columns={amenity_col: "amenity"}, inplace=True)
    if shop_col:
        # If 'shop' exists, rename it; if not, create an empty 'shop' column
        if shop_col in gdf_pois.columns:
            gdf_pois.rename(columns={shop_col: "shop"}, inplace=True)
        else:
            gdf_pois["shop"] = None

    # For land-use, rename that column to 'landuse'
    gdf_land.rename(columns={land_col: "landuse"}, inplace=True)

    return {"roads": gdf_roads, "pois": gdf_pois, "landuse": gdf_land}

# ─────────────────────────────────────────────────────────────────────────────
# Summarise road metrics per mesh cell
# ─────────────────────────────────────────────────────────────────────────────
def summarise_roads(gdf_roads: gpd.GeoDataFrame, mesh: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each mesh cell (identified by mesh.geom_id), compute:
      - road_len: total length of road within that cell (in metres)
      - road_share: fraction of this cell's road_len relative to the sum of all road length in the study area

    Steps:
      1. Reproject both roads and mesh to EPSG:3857 (metre-based) for accurate length calculations.
      2. Overlay roads and mesh polygons to get the intersection pieces.
      3. Compute the length of each intersected piece.
      4. Sum lengths per mesh.geom_id to obtain road_len.
      5. Divide each cell's road_len by the total to get road_share.

    Parameters
    ----------
    gdf_roads : GeoDataFrame
        Road segments with geometry in EPSG:4326.
    mesh : GeoDataFrame
        The empty mesh containing 'geom_id' and geometry, in EPSG:4326.

    Returns
    -------
    DataFrame
        Indexed by geom_id, with columns:
          - 'road_len' (metres)
          - 'road_share' (unitless fraction)
    """
    # 1) Reproject to EPSG:3857 (metre-based) for accurate length ops
    roads_m = gdf_roads.to_crs(3857)
    mesh_m = mesh.to_crs(3857)

    # 2) Compute intersection overlay: splits road segments by mesh polygons
    overlay = gpd.overlay(roads_m, mesh_m[["geom_id", "geometry"]], how="intersection")

    # 3) Compute each intersected segment's length in metres
    overlay["seg_len"] = overlay.geometry.length

    # 4) Group by geom_id and sum seg_len to get total road length per cell
    road_by_cell = overlay.groupby("geom_id")["seg_len"].sum().to_frame("road_len")

    # 5) Compute share relative to the overall sum
    total_len = road_by_cell["road_len"].sum()
    if total_len > 0:
        road_by_cell["road_share"] = road_by_cell["road_len"] / total_len
    else:
        # If no roads found, set share to 0
        road_by_cell["road_share"] = 0.0

    return road_by_cell.fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# Summarise POI metrics per mesh cell
# ─────────────────────────────────────────────────────────────────────────────
def summarise_pois(
    gdf_pois: gpd.GeoDataFrame,
    mesh: gpd.GeoDataFrame,
    relevant_poi: List[str]
) -> pd.DataFrame:
    """
    Filter POIs to keep only those whose 'amenity', 'shop', or 'fclass' value
    is in `relevant_poi`. Then count how many such POIs fall within each mesh cell,
    and compute the fraction relative to the total count.

    Steps:
      1. Create a boolean mask to filter POIs by checking 'amenity', 'shop', or 'fclass'.
      2. Spatially join filtered POIs to mesh polygons (point-in-polygon).
      3. Count the number of POIs per geom_id → 'poi_count'.
      4. Divide each cell's poi_count by the overall sum to get 'poi_share'.

    Parameters
    ----------
    gdf_pois : GeoDataFrame
        POI points (geometry type = Point) in EPSG:4326.
    mesh : GeoDataFrame
        The mesh containing 'geom_id' and geometry, in EPSG:4326.
    relevant_poi : list[str]
        List of tag values (e.g. ["supermarket", "hospital", ...]) to include.

    Returns
    -------
    DataFrame
        Indexed by geom_id, with columns:
          - 'poi_count' (integer)
          - 'poi_share' (unitless fraction)
    """
    # 1) Build a boolean mask for relevant POIs
    #    If 'amenity' or 'shop' columns exist, check them first; else use 'fclass'
    if {"amenity", "shop"} & set(gdf_pois.columns):
        mask = (
            gdf_pois.get("amenity", pd.Series(dtype=str)).isin(relevant_poi) |
            gdf_pois.get("shop",    pd.Series(dtype=str)).isin(relevant_poi)
        )
    else:
        # Fall back to matching on 'fclass' column
        mask = gdf_pois["fclass"].isin(relevant_poi)

    filtered = gdf_pois[mask].copy()

    # 2) Spatial join: assign each filtered POI to the mesh cell it falls within
    pois4326 = filtered.set_geometry("geometry").to_crs(4326)
    mesh4326 = mesh[["geom_id", "geometry"]].set_geometry("geometry").to_crs(4326)
    joined = gpd.sjoin(pois4326, mesh4326, how="inner", predicate="within")

    # 3) Count POIs per geom_id
    poi_counts = joined.groupby("geom_id").size().to_frame("poi_count")

    # 4) Compute poi_share by dividing by the total count
    total_poi = poi_counts["poi_count"].sum()
    if total_poi > 0:
        poi_counts["poi_share"] = poi_counts["poi_count"] / total_poi
    else:
        poi_counts["poi_share"] = 0.0

    return poi_counts.fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# Summarise land-use metrics per mesh cell
# ─────────────────────────────────────────────────────────────────────────────
def summarise_landuse(
    gdf_land: gpd.GeoDataFrame,
    mesh: gpd.GeoDataFrame,
    lu_classes: List[str]
) -> pd.DataFrame:
    """
    For each land-use class in `lu_classes`, compute:
      - 'lu_{class}_area': total area of that class within each mesh cell (m²)
      - 'lu_{class}_share': fraction of that class's area in this cell relative
                            to the total area of that class across all cells.

    Steps:
      1. Reproject both gdf_land and mesh to EPSG:3857 (metre-based) to compute areas.
      2. For each class:
         a. Subset polygons where 'landuse' == class.
         b. Overlay subset with mesh to get intersection pieces.
         c. Compute area of each piece.
         d. Sum area per geom_id → 'lu_{class}_area'.
         e. Divide by total area for that class to get 'lu_{class}_share'.
      3. Concatenate results for all classes into one DataFrame.

    Parameters
    ----------
    gdf_land : GeoDataFrame
        Land-use polygons with 'landuse' column, in EPSG:4326.
    mesh : GeoDataFrame
        The mesh containing 'geom_id' and geometry, in EPSG:4326.
    lu_classes : list[str]
        List of land-use class names of interest (e.g., ["industrial", "commercial", ...]).

    Returns
    -------
    DataFrame
        Indexed by geom_id, with columns for each class:
          - 'lu_{class}_area' (float)
          - 'lu_{class}_share' (float)
    """
    # 1) Reproject to metre-based CRS for accurate area calculations
    land_m = gdf_land.to_crs(3857)
    mesh_m = mesh.to_crs(3857)

    # Prepare a list to collect per-class DataFrames
    all_class_dfs = []

    for cls in lu_classes:
        # a) Subset land polygons of this class
        subset = land_m[land_m["landuse"] == cls]
        if subset.empty:
            # If no polygons of this class exist, skip
            continue

        # b) Overlay subset with the mesh to compute intersection pieces
        overlay = gpd.overlay(subset, mesh_m[["geom_id", "geometry"]], how="intersection")

        # c) Compute area of each intersected piece, in m²
        overlay["piece_area"] = overlay.geometry.area

        # d) Sum area per mesh cell (geom_id)
        area_by_cell = overlay.groupby("geom_id")["piece_area"].sum().to_frame(f"lu_{cls}_area")

        # e) Compute fraction relative to total area of this class
        total_area = area_by_cell[f"lu_{cls}_area"].sum()
        if total_area > 0:
            area_by_cell[f"lu_{cls}_share"] = area_by_cell[f"lu_{cls}_area"] / total_area
        else:
            area_by_cell[f"lu_{cls}_share"] = 0.0

        # f) Append this DataFrame to our list
        all_class_dfs.append(area_by_cell)

    # 2) If no land-use classes were found, return an empty DataFrame indexed by geom_id
    if not all_class_dfs:
        return pd.DataFrame(index=mesh["geom_id"])

    # 3) Concatenate all per-class results horizontally
    result = pd.concat(all_class_dfs, axis=1).fillna(0)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Merge static stats onto a mesh GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────
def enrich_mesh(base_mesh: gpd.GeoDataFrame, stats: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Left-join the stats DataFrame (indexed by geom_id) onto base_mesh, preserving
    geometry. Missing values are filled with zero.

    Parameters
    ----------
    base_mesh : GeoDataFrame
        Mesh containing columns 'geom_id' and geometry, in EPSG:4326.
    stats : DataFrame
        Indexed by geom_id, containing columns like 'road_len', 'poi_count', etc.

    Returns
    -------
    GeoDataFrame
        The enriched mesh, with all columns in `stats` appended.
    """
    # Perform a merge on the 'geom_id' column
    enriched = base_mesh.merge(
        stats.reset_index(),
        on="geom_id",
        how="left"
    ).fillna(0)

    return enriched

# ─────────────────────────────────────────────────────────────────────────────
# Batch process all mesh files for one city
# ─────────────────────────────────────────────────────────────────────────────
def batch_write(
    city: str,
    mesh_folder_in: Path,
    mesh_folder_out: Path,
    osm_shapefile: Path,
    relevant_poi: List[str],
    landuse_classes: List[str],
):
    """
    Read one sample mesh file to get 'geom_id' schema. Load OSM layers once, compute
    static road/poi/land-use metrics, then loop through each daily mesh file, merge
    those static stats, and write to mesh_folder_out (overwriting existing files).

    Additionally, saves a single enriched demo for date '2023-01-01' into
    data/demo-data for quick visual QA.

    Parameters
    ----------
    city : str
        City name (e.g., "addis" or "baghdad"), used for print statements.
    mesh_folder_in : Path
        Directory containing empty meshes (.gpkg), one per day.
    mesh_folder_out : Path
        Directory where enriched meshes will be written.
    osm_shapefile : Path
        Path to the OSM shapefile for this country/city.
    relevant_poi : list[str]
        List of POI tag values to count (e.g., ["supermarket", "hospital", ...]).
    landuse_classes : list[str]
        List of land-use categories to summarise (e.g., ["industrial", "commercial", ...]).

    Returns
    -------
    None
    """
    # 1) Read a sample mesh to get the base schema (geom_id + geometry)
    sample_file = next(mesh_folder_in.glob("*.gpkg"))
    base_mesh = gpd.read_file(str(sample_file))[["geom_id", "geometry"]].copy()

    # 2) Load OSM layers (roads, POIs, land-use)
    layers = load_osm_layers(osm_shapefile)
    roads_gdf = layers["roads"]
    pois_gdf = layers["pois"]
    landuse_gdf = layers["landuse"]

    # 3) Compute static metrics once:
    #    a) Roads metrics per mesh cell
    roads_stats = summarise_roads(roads_gdf, base_mesh)
    #    b) POI metrics per mesh cell
    pois_stats = summarise_pois(pois_gdf, base_mesh, relevant_poi)
    #    c) Land-use metrics per mesh cell
    land_stats = summarise_landuse(landuse_gdf, base_mesh, landuse_classes)

    # 4) Combine all metrics into one DataFrame, indexed by geom_id
    stats = roads_stats.join(pois_stats, how="outer").join(land_stats, how="outer").fillna(0)

    # 5) Save a single enriched demo for 2023-01-01 into data/demo-data
    #    (Assumes mesh filenames include "2023-01-01" exactly)
    demo_dir = mesh_folder_in.parent / "demo-data"
    demo_dir.mkdir(exist_ok=True)
    try:
        demo_mesh = gpd.read_file(str(demo_dir / f"{city}-2023-01-01.gpkg"))
    except FileNotFoundError:
        # Build the demo from the base_mesh + stats
        enriched_demo = enrich_mesh(base_mesh, stats)
        demo_fp = demo_dir / f"{city}-2023-01-01.gpkg"
        enriched_demo.to_file(str(demo_fp), driver="GPKG")
        print(f"[{city}] Saved demo mesh: {demo_fp.name}")

    # 6) Create output folder if it doesn't exist
    mesh_folder_out.mkdir(parents=True, exist_ok=True)

    # 7) Loop through every daily mesh file, merge static stats, and write out
    for mesh_file in sorted(mesh_folder_in.glob("*.gpkg")):
        # Read the daily empty mesh (geom_id + geometry)
        mesh_df = gpd.read_file(str(mesh_file))[["geom_id", "geometry"]].copy()
        # Merge static stats from earlier
        enriched = enrich_mesh(mesh_df, stats)
        # Write to output folder, overwriting if necessary
        out_fp = mesh_folder_out / mesh_file.name
        enriched.to_file(str(out_fp), driver="GPKG")
        print(f"[{city}] Wrote enriched mesh: {out_fp.name}")