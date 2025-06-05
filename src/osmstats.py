"""
OSM → Mesh enrichment utility
─────────────────────────────
This version handles both:
  • Raw OSM shapefiles that contain separate layers for roads, POIs, land‐use  
    (e.g. Geofabrik “ethiopia-latest-free” directory)
  • Single-layer OSM shapefiles that collapse many tags into 'fclass'.

We always:
  1. Read the correct sub-shapefiles (roads, pois, landuse) if `osm_shapefile` is a directory.
  2. Read a single-layer shapefile if `osm_shapefile` is a .shp file.
  3. Operate in EPSG:4326 for all geometry reads & writes.
  4. Temporarily project to EPSG:3857 to compute lengths (for roads) or areas (for land‐use).
  5. Return the final enriched mesh in EPSG:4326.
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
    raise ValueError(
        f"None of the columns {candidates} found in the supplied OSM layer. "
        f"Available columns: {list(gdf.columns)[:20]}…"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load OSM layers: roads, POIs, and land‐use
# ─────────────────────────────────────────────────────────────────────────────
def load_osm_layers(osm_shapefile: Path) -> dict:
    """
    Which file(s) we read depends on whether `osm_shapefile` is:
      • a directory (Geofabrik-style containing separate '*roads*.shp', '*poi*.shp', '*landuse*.shp')
      • a single `.shp` file that must be split by tags.

    Returns a dict with GeoDataFrames for "roads", "pois", and "landuse".
    """

    # If `osm_shapefile` is a directory, look for the three relevant sub-shapefiles:
    if osm_shapefile.is_dir():
        # 1) Find "*roads*.shp"
        roads_path = next(osm_shapefile.glob("*roads*.shp"), None)
        # 2) Find "*poi*.shp"  (Geofabrik uses "poi" or "pois")
        pois_path  = next(osm_shapefile.glob("*poi*.shp"), None)
        # 3) Find "*landuse*.shp"
        landuse_path = next(osm_shapefile.glob("*landuse*.shp"), None)

        if roads_path is None or pois_path is None or landuse_path is None:
            raise FileNotFoundError(
                f"Could not locate 'roads', 'poi', and 'landuse' .shp files under {osm_shapefile}.\n"
                f"Ensure the directory contains at least one '*roads*.shp', one '*poi*.shp', and one '*landuse*.shp'."
            )

        # Read each layer in EPSG:4326
        gdf_roads = gpd.read_file(str(roads_path)).to_crs("EPSG:4326")
        gdf_pois  = gpd.read_file(str(pois_path )).to_crs("EPSG:4326")
        gdf_land  = gpd.read_file(str(landuse_path)).to_crs("EPSG:4326")

        # ─── Harmonize POI columns ───────────────────────────────────────────────
        # Some POI shapefiles supply columns "amenity" or "shop" or both; else they collapse into "fclass".
        if "amenity" not in gdf_pois.columns:
            # If no "amenity", create empty amenity column
            gdf_pois["amenity"] = None
        if "shop" not in gdf_pois.columns:
            # If no "shop", create empty shop column
            gdf_pois["shop"] = None
        if "fclass" not in gdf_pois.columns:
            # If no "fclass" column, create an empty one
            gdf_pois["fclass"] = None

        # ─── Harmonize Landuse column ─────────────────────────────────────────────
        # If the landuse shapefile has "landuse", keep it. Otherwise
        # if it only has "fclass", rename that to "landuse".
        if "landuse" in gdf_land.columns:
            # Nothing to do
            pass
        elif "fclass" in gdf_land.columns:
            # Rename fclass → landuse
            gdf_land = gdf_land.rename(columns={"fclass": "landuse"})
        else:
            # Neither 'landuse' nor 'fclass' exist → error
            raise KeyError(
                f"The shapefile {landuse_path.name} must contain either a 'landuse' or 'fclass' column."
            )

        # Rename the roads attribute (often "highway" or "fclass") to "fclass"
        road_attr = _pick_column(gdf_roads, ["highway", "fclass"])
        if road_attr != "fclass":
            gdf_roads = gdf_roads.rename(columns={road_attr: "fclass"})

    else:
        # If `osm_shapefile` is a single .shp, we must split by tags in one layer:
        gdf = gpd.read_file(str(osm_shapefile)).to_crs("EPSG:4326")

        # Identify columns for roads vs. POIs vs. land-use
        road_attr    = _pick_column(gdf, ["highway", "fclass"])
        amenity_col  = "amenity" if "amenity" in gdf.columns else None
        shop_col     = "shop"    if "shop"    in gdf.columns else None
        land_attr    = _pick_column(gdf, ["landuse", "fclass"])

        # ─── Extract Roads ───────────────────────────────────────────────────────
        road_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
        gdf_roads = gdf[
            (gdf.geometry.type.isin(["LineString", "MultiLineString"])) &
            (gdf[road_attr].isin(road_classes))
        ].copy()
        # Rename whichever column we picked as "fclass"
        gdf_roads = gdf_roads.rename(columns={road_attr: "fclass"})

        # ─── Extract POIs ────────────────────────────────────────────────────────
        poi_mask = gdf.geometry.type.eq("Point")
        if amenity_col:
            poi_mask &= gdf[amenity_col].notna()
        elif shop_col:
            poi_mask &= gdf[shop_col].notna()
        else:
            poi_mask &= gdf[road_attr].notna()

        gdf_pois = gdf[poi_mask].copy()

        # Harmonize POI columns:
        if amenity_col:
            gdf_pois = gdf_pois.rename(columns={amenity_col: "amenity"})
        else:
            gdf_pois["amenity"] = None

        if shop_col:
            gdf_pois = gdf_pois.rename(columns={shop_col: "shop"})
        else:
            gdf_pois["shop"] = None

        if "fclass" not in gdf_pois.columns:
            # If we never had a "fclass" column, create an empty one
            gdf_pois["fclass"] = None

        # ─── Extract Land‐use Polygons ───────────────────────────────────────────
        landuse_mask = (
            gdf.geometry.type.isin(["Polygon", "MultiPolygon"]) &
            gdf[land_attr].notna()
        )
        gdf_land = gdf[landuse_mask].copy()

        # Rename the land‐use column to "landuse"
        if land_attr != "landuse":
            gdf_land = gdf_land.rename(columns={land_attr: "landuse"})

        # Finally, ensure roads/pois/landuse are all in EPSG:4326
        gdf_roads = gdf_roads.to_crs("EPSG:4326")
        gdf_pois  = gdf_pois.to_crs("EPSG:4326")
        gdf_land  = gdf_land.to_crs("EPSG:4326")

    # Return the three prepared GeoDataFrames
    return {"roads": gdf_roads, "pois": gdf_pois, "landuse": gdf_land}


# ─────────────────────────────────────────────────────────────────────────────
# Summarise road metrics per mesh cell
# ─────────────────────────────────────────────────────────────────────────────
def summarise_roads(gdf_roads: gpd.GeoDataFrame, mesh: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each mesh cell (mesh.geom_id), compute:
      • road_len:   total length of road within that cell (in metres)
      • road_share: cell_road_len / total_network_length

    Steps:
      1. Reproject both to EPSG:3857 to measure lengths in metres.
      2. Overlay to get clipped segments for each mesh cell.
      3. Compute segment lengths, sum by geom_id.
      4. Compute road_share.

    Returns a DataFrame indexed by geom_id with columns ['road_len','road_share'].
    """
    # 1) Reproject to metres
    roads_m = gdf_roads.to_crs("EPSG:3857")
    mesh_m  = mesh.to_crs("EPSG:3857")

    # 2) Intersection overlay: splits each road by the cell it falls in
    overlay = gpd.overlay(
        roads_m,
        mesh_m[["geom_id", "geometry"]],
        how="intersection"
    )

    # 3) Compute each clipped segment's length (m)
    overlay["seg_len"] = overlay.geometry.length

    # 4) Sum seg_len by geom_id → 'road_len'
    road_by_cell = overlay.groupby("geom_id")["seg_len"].sum().to_frame("road_len")

    # 5) Compute road_share = road_len / total_len
    total_len = float(road_by_cell["road_len"].sum())
    if total_len > 0:
        road_by_cell["road_share"] = road_by_cell["road_len"] / total_len
    else:
        road_by_cell["road_share"] = 0.0

    return road_by_cell.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# Summarise POI metrics per mesh cell
# ─────────────────────────────────────────────────────────────────────────────
def summarise_pois(
    mesh:     gpd.GeoDataFrame,
    gdf_pois: gpd.GeoDataFrame,
    relevant_poi: list[str],
) -> pd.DataFrame:
    """
    For each mesh cell (mesh.geom_id), compute:
      • poi_count: number of relevant POIs within that cell
      • poi_share: poi_count / total_relevant_poi_count

    Steps:
      1. Build a unified boolean mask across 'amenity', 'shop' and 'fclass' columns
         (only considering non-null values in those columns).
      2. Spatially join filtered POIs → mesh (predicate="within").
      3. Count by geom_id, compute poi_share.

    Returns a DataFrame indexed by geom_id with columns ['poi_count','poi_share'].
    """
    # 1) Build unified boolean mask for all three tag‐columns:
    cols_to_check = []
    if "amenity" in gdf_pois.columns:
        cols_to_check.append("amenity")
    if "shop" in gdf_pois.columns:
        cols_to_check.append("shop")
    if "fclass" in gdf_pois.columns:
        cols_to_check.append("fclass")

    # If none of these columns exist, raise an error:
    if not cols_to_check:
        raise KeyError(
            "summarise_pois: POI GeoDataFrame must contain at least one of "
            "'amenity', 'shop', or 'fclass' columns."
        )

    # Create boolean DataFrame where each column tests membership in relevant_poi.
    mask_df = pd.DataFrame(
        { col: gdf_pois[col].fillna("").isin(relevant_poi) for col in cols_to_check }
    )

    # Collapse across all tag‐columns: if ANY column matches, keep that POI.
    mask = mask_df.any(axis=1)
    filtered = gdf_pois[mask].copy()

    # Early exit: if there are zero relevant POIs, return a zero‐filled DataFrame for all geom_id
    if len(filtered) == 0:
        empty_stats = pd.DataFrame(
            {
                "geom_id": mesh["geom_id"].values,
                "poi_count": 0,
                "poi_share": 0.0,
            }
        ).set_index("geom_id")
        return empty_stats

    # 2) Spatial join (point‐in‐polygon) to assign each POI → geom_id
    pois4326 = filtered.to_crs("EPSG:4326")
    mesh4326 = mesh[["geom_id", "geometry"]].set_geometry("geometry").to_crs("EPSG:4326")

    joined = gpd.sjoin(
        pois4326,
        mesh4326,
        how="inner",
        predicate="within",
    )

    # 3) Compute per‐cell counts; poi_share = count / total relevant POIs
    counts = joined.groupby("geom_id").size().rename("poi_count").to_frame()
    total_relevant = len(filtered)
    counts["poi_share"] = counts["poi_count"] / float(total_relevant)

    # 4) Reindex so every mesh cell appears (fill missing → zero)
    stats = counts.reindex(mesh["geom_id"], fill_value=0)
    stats = stats.astype({"poi_count": "int64", "poi_share": "float64"})

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Summarise land‐use metrics per mesh cell
# ─────────────────────────────────────────────────────────────────────────────
def summarise_landuse(
    gdf_land: gpd.GeoDataFrame,
    mesh: gpd.GeoDataFrame,
    lu_classes: List[str]
) -> pd.DataFrame:
    """
    For each land‐use class in `lu_classes`, compute per mesh cell:
      • lu_{class}_area  (total area in m²)
      • lu_{class}_share (share = area_cell / total_area_class)

    Steps:
      1. Reproject `gdf_land` & `mesh` to EPSG:3857 for area computations.
      2. For each class:
         a. Subset polygons where gdf_land["landuse"] == cls.
         b. Overlay subset with mesh_m to get clipped pieces.
         c. Compute piece_area = geometry.area.
         d. Sum piece_area by geom_id → lu_{cls}_area.
         e. Compute lu_{cls}_share = lu_{cls}_area / total_area_cls.
      3. Concatenate all class results horizontally.

    Returns a DataFrame indexed by geom_id, with columns:
      ['lu_{cls}_area', 'lu_{cls}_share', ...] for each class in lu_classes.
    """
    # 1) Reproject to metres
    land_m = gdf_land.to_crs("EPSG:3857")
    mesh_m = mesh.to_crs("EPSG:3857")

    metrics_list = []

    for cls in lu_classes:
        # a) Subset polygons of class `cls`
        subset = land_m[land_m["landuse"] == cls]
        if subset.empty:
            continue

        # b) Overlay clipped polys with mesh to get intersection
        overlay = gpd.overlay(
            subset,
            mesh_m[["geom_id", "geometry"]],
            how="intersection"
        )

        # c) Compute area (m²) of each piece
        overlay["piece_area"] = overlay.geometry.area

        # d) Sum up piece_area by geom_id → 'lu_{cls}_area'
        area_by_cell = overlay.groupby("geom_id")["piece_area"].sum().to_frame(f"lu_{cls}_area")

        # e) Compute 'lu_{cls}_share'
        total_area = float(area_by_cell[f"lu_{cls}_area"].sum())
        if total_area > 0:
            area_by_cell[f"lu_{cls}_share"] = area_by_cell[f"lu_{cls}_area"] / total_area
        else:
            area_by_cell[f"lu_{cls}_share"] = 0.0

        metrics_list.append(area_by_cell)

    # If nothing matched, return an empty DataFrame (so that merge still works)
    if not metrics_list:
        return pd.DataFrame(index=mesh["geom_id"])

    # Concatenate all class‐by‐cell results horizontally
    result = pd.concat(metrics_list, axis=1).fillna(0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Merge all static stats onto a mesh GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────
def enrich_mesh(base_mesh: gpd.GeoDataFrame, stats: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Left‐join `stats` (indexed by geom_id) onto `base_mesh`, preserving geometry.
    Fill missing values with zero.

    Returns the enriched mesh (EPSG:4326).
    """
    enriched = base_mesh.merge(
        stats.reset_index(),
        on="geom_id",
        how="left"
    ).fillna(0)

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Batch‐process all mesh files for one city
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
    1) Read a single 'sample' mesh (to get the 'geom_id' + 'geometry' schema).
    2) Load OSM layers (roads, pois, landuse) exactly once.
    3) Compute static metrics for:
         a) roads   → summarise_roads()
         b) pois    → summarise_pois()
         c) landuse → summarise_landuse()
    4) Save a one‐off “2023-01-01” demo mesh into data/demo-data/.
    5) Loop over all daily mesh files in mesh_folder_in:
         • Merge the static metrics into each empty mesh.
         • Write the enriched mesh (EPSG:4326) to mesh_folder_out.

    Parameters
    ----------
    city : str
        “addis” or “baghdad” (used in print statements).
    mesh_folder_in : Path
        Directory of empty-mesh GPKGs (one per date).
    mesh_folder_out : Path
        Directory where enriched-mesh GPKGs will be written.
    osm_shapefile : Path
        Either a directory (Geofabrik style) containing `*roads*.shp`, `*poi*.shp`, `*landuse*.shp`
        or a single .shp that must be split internally.
    relevant_poi : list[str]
        e.g. ["supermarket", "hospital", …].
    landuse_classes : list[str]
        e.g. ["industrial", "commercial", "farmland", "farmyard", "residential", "retail"].

    Returns
    -------
    None
    """
    # 1) Read a “sample” mesh to fix the schema (geom_id + geometry)
    sample_file = next(mesh_folder_in.glob("*.gpkg"))
    base_mesh = gpd.read_file(str(sample_file))[["geom_id", "geometry"]].copy()
    base_mesh = base_mesh.to_crs("EPSG:4326")

    # 2) Load OSM layers (roads, pois, landuse)
    layers = load_osm_layers(osm_shapefile)
    roads_gdf   = layers["roads"]   # already in EPSG:4326
    pois_gdf    = layers["pois"]    # already in EPSG:4326
    landuse_gdf = layers["landuse"] # already in EPSG:4326

    # 3) Compute static metrics once:
    roads_stats = summarise_roads(roads_gdf, base_mesh)
    pois_stats  = summarise_pois(base_mesh, pois_gdf, relevant_poi)
    land_stats  = summarise_landuse(landuse_gdf, base_mesh, landuse_classes)

    # 4) Combine into one DataFrame (indexed by geom_id)
    stats = roads_stats.join(pois_stats, how="outer").join(land_stats, how="outer").fillna(0)

    # 5) Save a single “2023-01-01” demo mesh (if it does not already exist)
    demo_dir = mesh_folder_in.parent / "demo-data"
    demo_dir.mkdir(exist_ok=True)
    demo_fp = demo_dir / f"{city}-2023-01-01.gpkg"
    if not demo_fp.exists():
        enriched_demo = enrich_mesh(base_mesh, stats)
        enriched_demo.to_file(str(demo_fp), driver="GPKG")
        print(f"[{city}]  ✓ Saved demo mesh: {demo_fp.name}")

    # 6) Ensure the output folder exists
    mesh_folder_out.mkdir(parents=True, exist_ok=True)

    # 7) Loop over every daily mesh file, enrich and write out
    for mesh_file in sorted(mesh_folder_in.glob("*.gpkg")):
        mesh_df = gpd.read_file(str(mesh_file))[["geom_id", "geometry"]].copy()
        mesh_df = mesh_df.to_crs("EPSG:4326")

        enriched = enrich_mesh(mesh_df, stats)

        out_fp = mesh_folder_out / mesh_file.name
        enriched.to_file(str(out_fp), driver="GPKG")
        print(f"[{city}]  ✓ Wrote enriched mesh: {out_fp.name}")