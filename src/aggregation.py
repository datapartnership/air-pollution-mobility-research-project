import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import fiona
import rasterio
import os
import numpy as np
from pathlib import Path
import geopandas as gpd
from rasterstats import zonal_stats
from pathlib import Path
import pandas as pd
from pandas import DataFrame

# Function: Aggregate data within each mesh
def aggregate_to_mesh(mesh_grid, 
                      tiff_path:Path,
                      feature_name: str)-> gpd.GeoDataFrame:
    """
    Aggregated a single tiff file under tiff_path

    Parameters:
    ----------
    mesh_grid: GeoPackage file
        The mesh gird of the interested area.
    tiff_path:
        The path of the to be processed tiff file.

    Output:
    ------
        Return the aggregated data in the mesh (GeoPackge format)
    
    """

    stats = zonal_stats(mesh_grid, tiff_path, stats=["mean"], nodata=np.nan)
    mesh_grid[feature_name] = [s["mean"] for s in stats]

    return mesh_grid

# Function: Aggregate and write aggregated values to multiple meshes
def aggregate_data(
        data_tiff_path:Path, 
        mesh_path: Path,
        feature_name: str,
        layer_name: str)-> None :
    """
    Aggregate all the tiff files under the data_tiff_path to mesh. Multiple output files.

    Parameters
    ----------
    data_tiff_path: Path
        Path of the tiff files to be processed.
    mesh_path: Path
        Directory that will save the output files.

    Output
    --------
    Return tiff files under a new created folder in data/countryname_NO2_filled
        
    """
    
    filled_tiffs = [f for f in os.listdir(data_tiff_path) if f.lower().endswith('.tif')]
    abs_filled_paths = [data_tiff_path / f for f in filled_tiffs] # absolute path
    n_task = len(filled_tiffs)
    
    for index, tiff_path in enumerate(abs_filled_paths):

        date = filled_tiffs[index].split('_')[2].split('.')[0]
        print(f"currently working on: {index+1}/{n_task}, {date}")

        matched_mesh =  [f for f in mesh_path.glob("*.gpkg") if date in f.name][0]  # typically only one match， absolute path
        mesh = gpd.read_file(matched_mesh)

        try:
            new_mesh = aggregate_to_mesh(mesh, tiff_path, feature_name)
            new_mesh.to_file(matched_mesh, driver='GPKG', layer=layer_name, mode='w')


        except Exception as e:
            print(f"Error processing {tiff_path} at index {index}: {e}")
            print("Skipping to next file...")
            continue
        


def aggregate_pop_data(
        data_tiff_path: Path, 
        mesh_path: Path,
        layer_name: str,
        output_path: Path,
        agg_type: str = "sum",
        feature_col: str = "pop_sum_m",
        
):
    """
    Aggregate all the tiff files under the data_tiff_path to mesh. Multiple output files.

    Parameters
    ----------
    data_tiff_path: Path
        Path of the tiff files to be processed.
    mesh_path: Path
        Path to the mesh file (GeoPackage or GeoJSON).
    layer_name: str
        Name of the GPKG layer to save.
    agg_type: str
        Aggregation type (e.g., 'sum', 'mean', etc.)
    feature_col: str
        Name of the new column storing the aggregation result.
    output_path: Path
        Directory to save the output .gpkg files.

    Output
    ------
    Saves GPKG files with aggregated data into the output_path directory.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    filled_tiffs = [f for f in os.listdir(data_tiff_path) if f.lower().endswith('.tif')]
    abs_filled_paths = [data_tiff_path / f for f in filled_tiffs] 
    n_task = len(filled_tiffs)
    
    for index, tiff_path in enumerate(abs_filled_paths):
        year = filled_tiffs[index].split('_')[2].split('.')[0]  # example: pop_eth_2020.tif
        print(f"Currently working on: {index + 1}/{n_task}, Year: {year}")

        mesh_grid = gpd.read_file(mesh_path)

        try:
            with rasterio.open(tiff_path) as src:
                nodata_val = src.nodata or -99999.0
            stats = zonal_stats(mesh_grid, tiff_path, stats=[agg_type], nodata=nodata_val)
            mesh_grid[feature_col] = [max(0, s[agg_type] or 0) for s in stats]

            # Set output file path
            output_file = output_path / f"pop_aggregated_{year}.gpkg"

            # Save to GPKG
            mesh_grid.to_file(output_file, driver='GPKG', layer=layer_name, mode='w')

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error processing {tiff_path} at index {index}: {e}")
            print("Skipping to next file...")
            continue

from typing import List
def compute_mean_mesh_by_daytype(
    date_df: DataFrame,
    meshes_path: Path,
    output_path: Path,
    feature_col: str = "no2_mean",
    output_name: str = "mean_mesh_workdays_weekends.gpkg",
    country: str = "Ethiopia",
    city: str = "addis-ababa",
    grid_id_col: str = "geom_id",
    specify_date: bool = False,
    selected_dates: List = ["2023-01-01", "2023-01-02"],
) -> None:

    """
    Compute the mean value of a specified feature over spatial grids, grouped by day types 
    (e.g., Workdays, Weekends), and save the merged result to a GeoPackage file.

    Parameters
    ----------
    date_df : pandas.DataFrame
        DataFrame containing at least two columns: 'Date' (datetime) and 
        '<country>_Workday_Type', which defines the grouping (e.g., "Workdays", "Weekends").

    meshes_path : pathlib.Path
        Path to the directory containing input GPKG files (one per day), named like 'city-YYYY-MM-DD.gpkg'.

    output_path : pathlib.Path
        Directory where the output GeoPackage will be saved.

    feature_col : str, optional (default="no2_mean")
        Name of the numeric column in GPKG files to average across days.

    output_name : str, optional (default="mean_mesh_workdays_weekends.gpkg")
        Name of the output GeoPackage file.

    country : str, optional (default="Ethiopia")
        Country name used to select the correct '<country>_Workday_Type' column in `date_df`.

    city : str, optional (default="addis-ababa")
        Used to parse dates from GPKG filenames, assumed to follow the format 'city-YYYY-MM-DD.gpkg'.

    grid_id_col : str, optional (default="geom_id")
        Column name that uniquely identifies each spatial grid cell.

    Returns
    -------
    None
        Saves the computed mean GeoDataFrame as a GPKG file to `output_path / output_name`.
    """

    # Format date strings and map them to workday/weekend types
    date_df["Date_str"] = date_df["Date"].dt.strftime("%Y-%m-%d")
    date_class_map = dict(zip(date_df["Date_str"], date_df[f"{country}_Workday_Type"]))

    # Scan all GPKG files
    file_paths = sorted(meshes_path.glob("*.gpkg"))

    # Prepare containers for each unique day class (e.g., Workdays, Weekends)
    day_classes = set(date_df[f"{country}_Workday_Type"])
    group_frames = {key: [] for key in day_classes}

    # Process each file and group by date class
    for fp in file_paths:
        # Extract date string from filename
        date_str = os.path.basename(fp).split(f"{city}-")[-1].split(".gpkg")[0]

        # Get corresponding day type from lookup
        group_key = date_class_map.get(date_str)
        if group_key not in group_frames:
            raise KeyError(f"Date '{date_str}' not found in date table or has an unknown class.")

        # Read GPKG file
        gdf = gpd.read_file(fp)

        # Skip if target feature column is missing
        if feature_col not in gdf.columns:
            continue

        if specify_date and date_str not in selected_dates:
            continue  # if specify date is true, check if the date is within the dates

        # Keep only required columns
        gdf = gdf[[grid_id_col, feature_col, "geometry"]]

        # Append to the relevant day group
        group_frames[group_key].append(gdf)

    # Compute mean per group
    mean_frames = {}
    for group_key, frames in group_frames.items():
        if not frames:
            raise ValueError(f"No files found for group '{group_key}'!")

        # Stack all daily grids
        stacked = pd.concat(frames, ignore_index=True)

        # Group by grid ID and compute mean
        mean_df = (
            stacked.groupby(grid_id_col, as_index=False)
                   .agg({feature_col: "mean"})
                   .rename(columns={feature_col: f"{group_key}_mean"})
        )

        # Use geometry from the first frame
        geom_ref = frames[0][[grid_id_col, "geometry"]]
        mean_gdf = gpd.GeoDataFrame(
            mean_df.merge(geom_ref, on=grid_id_col, how="left"),
            geometry="geometry",
            crs=frames[0].crs
        )

        mean_frames[group_key] = mean_gdf

    # Merge all groups side by side
    group_keys = list(mean_frames.keys())
    final_gdf = mean_frames[group_keys[0]]

    for key in group_keys[1:]:
        final_gdf = final_gdf.merge(
            mean_frames[key].drop(columns="geometry"),
            on=grid_id_col,
            how="left"
        )

    # Ensure output path exists
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / output_name

    # Save result to GPKG
    final_gdf.to_file(output_file, layer="mean_mesh", driver="GPKG")
    print(f"Saved mean meshes to {output_file}")

    return final_gdf

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import rasterio.plot

def aggregate_cloud_to_mesh(mesh_gpkg_path: Path, 
                            tiff_path: Path,
                            feature_name: str) -> gpd.GeoDataFrame:
    """
    Aggregates cloud state (0–3) using majority value per mesh cell.
    """
    import rasterio

    # 1. Load mesh
    mesh_grid = gpd.read_file(mesh_gpkg_path)

    # 2. Check nodata value
    with rasterio.open(tiff_path) as src:
        nodata_val = src.nodata if src.nodata is not None else 255  # 默认假设 255 为 nodata

    # 3. Run zonal stats
    stats = zonal_stats(
        vectors=mesh_grid,
        raster=str(tiff_path),
        categorical=True,
        nodata=nodata_val,
        all_touched=True,
        geojson_out=False
    )

    # 4. Get majority value
    def get_majority(cat_counts):
        if not cat_counts or not isinstance(cat_counts, dict):
            return np.nan
        return max(cat_counts.items(), key=lambda kv: kv[1])[0]

    mesh_grid[feature_name] = [get_majority(s) for s in stats]
    print(mesh_grid[[feature_name]].value_counts(dropna=False))

    return mesh_grid






import re

def aggregate_cloud_data(
        data_tiff_path: Path, 
        mesh_path: Path,
        feature_name: str,
        layer_name: str
    ) -> None:
    """
    For each cloud-state TIFF, find the matching mesh GPKG (by date in filename),
    aggregate by majority to that mesh, and overwrite the mesh layer.

    Parameters
    ----------
    data_tiff_path : Path
        Directory containing cloud-state .tif files (e.g. baghdad_cloud_2023-01-01_filled.tif).
    mesh_path : Path
        Directory containing mesh GPKG files (e.g. baghdad-2023-01-01.gpkg).
    feature_name : str
        Column name to store the aggregated cloud state.
    layer_name : str
        Layer name when writing back to GPKG.
    """
    tiff_files = sorted(f for f in data_tiff_path.glob("*.tif"))
    n_task = len(tiff_files)

    for idx, tiff_path in enumerate(tiff_files, 1):
        # Extract YYYY-MM-DD from TIFF filename
        m = re.search(r"\d{4}-\d{2}-\d{2}", tiff_path.name)
        if not m:
            print(f"[{idx}/{n_task}] ⚠ Cannot extract date from {tiff_path.name}, skipping")
            continue
        date_str = m.group(0)
        print(f"[{idx}/{n_task}] Aggregating cloud state: {date_str}")

        # Find the matching mesh GPKG
        meshes = list(mesh_path.glob(f"*{date_str}*.gpkg"))
        if not meshes:
            print(f"No matching mesh found for {date_str}, skipping")
            continue
        mesh_gpkg = meshes[0]

        try:
            # Generate GeoDataFrame with new column
            new_mesh = aggregate_cloud_to_mesh(mesh_gpkg, tiff_path, feature_name)

            # Write back to the same GPKG (overwrite or create layer)
            new_mesh.to_file(mesh_gpkg, driver="GPKG", layer=layer_name, mode="w")
            print(f"Layer '{layer_name}' written to {mesh_gpkg.name}")
        except Exception as err:
            print(f"Processing failed: {err}")
            continue