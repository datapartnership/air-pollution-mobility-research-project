import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import os
from shapely.ops import unary_union

# ------------------------------------------------------------
# CORE FUNCTIONS FOR OSM → MESH ATTRIBUTES
# ------------------------------------------------------------

def load_osm_layers(osm_shapefile: Path) -> dict:
    """
    Load the single Geofabrik shapefile into separate GeoDataFrames for:
      - roads (filtered by highway tag)
      - pois  (all points; filtering done downstream)
      - landuse (filtered by landuse tag)
    """
    # 1) Read full OSM shapefile once
    gdf = gpd.read_file(str(osm_shapefile))
    # Ensure CRS is EPSG:4326
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # 2) Partition into three themes
    #    a) Roads: highway in specific classes
    road_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
    roads = gdf[gdf["highway"].isin(road_classes)].copy()
    #    b) POIs: keep everything with a valid geometry and amenity/shop keys
    pois = gdf[(gdf.geometry.type == "Point") & 
               (gdf["amenity"].notna() | gdf["shop"].notna())].copy()
    #    c) Land-use: keep only polygons with a landuse key
    landuse = gdf[gdf["landuse"].notna() & (gdf.geometry.type.isin(["Polygon", "MultiPolygon"]))].copy()

    return {"roads": roads, "pois": pois, "landuse": landuse}


def summarise_roads(gdf_roads: gpd.GeoDataFrame, mesh: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each mesh cell (mesh.geom_id), compute:
      - road_len: total road length (in metres) within that cell
      - road_share: road_len / (sum of all road_len across study area)
    """

    # 1) Reproject to a metre-based CRS for accurate length (EPSG:3857)
    roads_m = gdf_roads.to_crs("EPSG:3857")
    mesh_m = mesh.to_crs("EPSG:3857")

    # 2) Spatial overlay: which road-segment pieces fall into each mesh
    overlay = gpd.overlay(roads_m, mesh_m[["geom_id", "geometry"]], how="intersection")
    # 3) Compute each intersected segment's length in metres
    overlay["seg_len"] = overlay.geometry.length

    # 4) Sum length per geom_id
    road_by_cell = overlay.groupby("geom_id")["seg_len"].sum().reset_index()
    road_by_cell.rename(columns={"seg_len": "road_len"}, inplace=True)

    # 5) Compute overall sum for share
    total_len = road_by_cell["road_len"].sum()
    if total_len > 0:
        road_by_cell["road_share"] = road_by_cell["road_len"] / total_len
    else:
        road_by_cell["road_share"] = 0.0

    return road_by_cell.set_index("geom_id")


def summarise_pois(
    gdf_pois: gpd.GeoDataFrame,
    mesh: gpd.GeoDataFrame,
    relevant_poi: list
) -> pd.DataFrame:
    """
    For each mesh cell, count only the POIs whose 'amenity' or 'shop' tag
    matches one of `relevant_poi`. Then compute:
      - poi_count: count of those POIs in each cell
      - poi_share: poi_count / (sum of all relevant POIs across study area)
    """
    # 1) Filter pois by relevant tag-values
    #    We look in both 'amenity' and 'shop' columns
    mask_amenity = gdf_pois["amenity"].isin(relevant_poi)
    mask_shop    = gdf_pois["shop"].isin(relevant_poi)
    pois_filt = gdf_pois[mask_amenity | mask_shop].copy()

    # 2) Spatial join: assign each point to a mesh cell
    pois_filt = pois_filt.set_geometry("geometry").to_crs("EPSG:4326")
    mesh_xy = mesh[["geom_id", "geometry"]].set_geometry("geometry").to_crs("EPSG:4326")
    joined = gpd.sjoin(pois_filt, mesh_xy, how="inner", predicate="within")

    # 3) Count per geom_id
    poi_by_cell = joined.groupby("geom_id").size().reset_index(name="poi_count")

    # 4) Compute share
    total_poi = poi_by_cell["poi_count"].sum()
    if total_poi > 0:
        poi_by_cell["poi_share"] = poi_by_cell["poi_count"] / total_poi
    else:
        poi_by_cell["poi_share"] = 0.0

    return poi_by_cell.set_index("geom_id")


def summarise_landuse(
    gdf_land: gpd.GeoDataFrame,
    mesh: gpd.GeoDataFrame,
    lu_classes: list
) -> pd.DataFrame:
    """
    For each mesh cell and each landuse class in lu_classes, compute:
      - lu_{class}_area: total area (m2) of that landuse type within cell
      - lu_{class}_share: (area in this cell) / (sum of that class's area across study area)
    Returns a DataFrame with index geom_id and one column per metric.
    """
    # 1) Reproject both to metre-based CRS (EPSG:3857)
    land_m = gdf_land.to_crs("EPSG:3857")
    mesh_m = mesh.to_crs("EPSG:3857")

    # 2) Keep only rows whose 'landuse' is one of the target classes
    land_m = land_m[land_m["landuse"].isin(lu_classes)].copy()

    # 3) Initialize a DataFrame to collect metrics
    metrics = {}

    # 4) For each class, subset and overlay
    for cls in lu_classes:
        subset = land_m[land_m["landuse"] == cls]
        if subset.empty:
            # Create zero-filled series for all cells
            zero_df = pd.DataFrame({
                f"lu_{cls}_area": np.zeros(len(mesh_m)),
                f"lu_{cls}_share": np.zeros(len(mesh_m))
            }, index=mesh_m["geom_id"])
            metrics[cls] = zero_df
            continue

        # a) Intersection overlay with mesh
        ov = gpd.overlay(subset, mesh_m[["geom_id", "geometry"]], how="intersection")
        # b) Compute area per intersected piece
        ov["piece_area"] = ov.geometry.area

        # c) Sum piece_area by geom_id
        area_by_cell = ov.groupby("geom_id")["piece_area"].sum().reset_index()
        area_by_cell.rename(columns={"piece_area": f"lu_{cls}_area"}, inplace=True)

        # d) Compute share: divide by total of class-area
        total_area = area_by_cell[f"lu_{cls}_area"].sum()
        if total_area > 0:
            area_by_cell[f"lu_{cls}_share"] = area_by_cell[f"lu_{cls}_area"] / total_area
        else:
            area_by_cell[f"lu_{cls}_share"] = 0.0

        # e) Pivot to full index of mesh IDs
        full = pd.DataFrame(index=mesh_m["geom_id"])
        full = full.join(area_by_cell.set_index("geom_id"), how="left").fillna(0)
        metrics[cls] = full

    # 5) Concatenate all metrics horizontally
    all_metrics = pd.concat([metrics[cls] for cls in lu_classes], axis=1)
    return all_metrics


def enrich_mesh(base_mesh: gpd.GeoDataFrame, stats: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Given a base_mesh GeoDataFrame (with geom_id) and a stats DataFrame
    (indexed by geom_id with one or more columns), perform a left-join
    of stats onto base_mesh. Fill missing values with zeros.
    """
    # 1) Ensure same CRS
    if base_mesh.crs.to_epsg() != 4326:
        base_mesh = base_mesh.to_crs("EPSG:4326")

    # 2) Merge
    enriched = base_mesh.merge(
        stats.reset_index(),
        on="geom_id",
        how="left"
    ).fillna(0)

    return enriched


def batch_write(
    city: str,
    mesh_folder_in: Path,
    mesh_folder_out: Path,
    osm_shapefile: Path,
    relevant_poi: list,
    landuse_classes: list
):
    """
    For a given city, read all base meshes from mesh_folder_in (one .gpkg per day),
    compute static OSM attributes once, then apply them to each daily file and write
    to mesh_folder_out (overwrite existing).
    """

    # 1) Load all empty meshes into a single GeoDataFrame—for schema & ID extraction
    sample_file = next(mesh_folder_in.glob("*.gpkg"))
    base_mesh = gpd.read_file(str(sample_file))
    # Keep only geom_id & geometry
    base_mesh = base_mesh[["geom_id", "geometry"]].copy()

    # 2) Load OSM
    layers = load_osm_layers(osm_shapefile)
    roads_gdf   = layers["roads"]
    pois_gdf    = layers["pois"]
    landuse_gdf = layers["landuse"]

    # 3) Summarise static metrics (once)
    roads_stats = summarise_roads(roads_gdf, base_mesh)
    pois_stats  = summarise_pois(pois_gdf, base_mesh, relevant_poi)
    land_stats  = summarise_landuse(landuse_gdf, base_mesh, landuse_classes)

    # 4) Combine all metrics into one DataFrame, indexed by geom_id
    all_stats = roads_stats.join(pois_stats, how="outer").join(land_stats, how="outer").fillna(0)

    # 5) Create enriched prototype mesh (for a single day) for demo
    enriched_sample = enrich_mesh(base_mesh, all_stats)

    # 6) Save the 2023-01-01 sample to data/demo-data
    demo_dir = mesh_folder_in.parent / "demo-data"
    demo_dir.mkdir(exist_ok=True)
    sample_name = f"{city.lower()}-2023-01-01.gpkg"
    enriched_sample.to_file(str(demo_dir / sample_name), driver="GPKG")

    # 7) Iterate through each daily mesh, join static attrs, and write
    mesh_folder_out.mkdir(parents=True, exist_ok=True)
    for mesh_path in sorted(mesh_folder_in.glob("*.gpkg")):
        mesh_date = mesh_path.stem  # e.g. "addis-ababa-2023-01-02"
        mesh_df = gpd.read_file(str(mesh_path))[["geom_id", "geometry"]]
        enriched = enrich_mesh(mesh_df, all_stats)
        out_fp = mesh_folder_out / f"{mesh_date}.gpkg"
        enriched.to_file(str(out_fp), driver="GPKG")
        print(f"[{city}] Wrote enriched mesh: {out_fp.name}")
