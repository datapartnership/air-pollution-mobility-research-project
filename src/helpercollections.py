import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
from pathlib import Path
import os

def merge_multiple_gpkgs(
        feature_mesh_paths: list[Path], 
        output_folder: Path
) -> None:
    """
    Merge multiple GeoPackage files from different folders by matching filenames,
    concatenating all layers within each file, and combining data side-by-side
    while avoiding duplicated columns.

    Parameters
    -----------
    feature_mesh_paths : list of Path
        List of folder paths, each containing GeoPackage files to merge.
        It is assumed that files with the same name exist in these folders.

    output_folder : Path
        Folder path where the merged GeoPackage files will be saved.
        This folder will be created if it doesn't exist.

    Example usage
    --------------
    `DATA_PATH = Path("your/data/root/path")`

    `feature_mesh_paths = [DATA_PATH / "addis-no2-mesh-data", DATA_PATH / "addis-OSM-mesh-data", DATA_PATH / "addis-pop-mesh-data"]`

    `output_folder = DATA_PATH / "addis-mesh-data"`

    `output_folder.mkdir(exist_ok=True)`
    
    `merge_multiple_gpkgs(feature_mesh_paths, output_folder)`
        
    """

    # Make sure the output folder exists
    output_folder.mkdir(exist_ok=True)
    
    # Get list of file names from the first directory (assume same files in all)
    file_names = [f.name for f in feature_mesh_paths[0].glob("*.gpkg")]

    for file_name in file_names:
        gdf_list = []

        for folder in feature_mesh_paths:
            file_path = folder / file_name

            if not file_path.exists():
                print(f"{file_path} does not exist, skipping.")
                continue

            # List all layers in the GeoPackage and read them
            layers = fiona.listlayers(file_path)
            layer_gdfs = []
            for lyr in layers:
                gdf = gpd.read_file(file_path, layer=lyr)
                layer_gdfs.append(gdf)

            if not layer_gdfs:
                print(f"No valid layers found in {file_path}, skipping.")
                continue

            # Concatenate all layers vertically into a single GeoDataFrame
            merged = pd.concat(layer_gdfs, ignore_index=True)
            merged_gdf = gpd.GeoDataFrame(merged)

            # Add 'no2_mean' column if it does not exist
            if 'no2_mean' not in merged_gdf.columns:
                merged_gdf['no2_mean'] = np.nan

            gdf_list.append(merged_gdf)

        if gdf_list:
            try:
                # Keep geometry from the first GeoDataFrame, drop geometry from others
                base_gdf = gdf_list[0]
                other_gdfs = []

                existing_columns = set(base_gdf.columns)
                existing_columns.discard('geometry')  # Handle geometry separately

                for gdf in gdf_list[1:]:
                    # Drop geometry column
                    gdf = gdf.drop(columns='geometry', errors='ignore')

                    # Remove columns that already exist in base_gdf to avoid duplicates
                    duplicate_cols = [col for col in gdf.columns if col in existing_columns]
                    gdf = gdf.drop(columns=duplicate_cols, errors='ignore')

                    # Update existing columns set
                    existing_columns.update(gdf.columns)

                    other_gdfs.append(gdf)

                # Concatenate all GeoDataFrames side-by-side (axis=1)
                combined_df = pd.concat([base_gdf] + other_gdfs, axis=1)
                combined_gdf = gpd.GeoDataFrame(combined_df, geometry='geometry', crs=base_gdf.crs)

                # Save the combined GeoDataFrame as a new GeoPackage
                output_path = output_folder / file_name
                combined_gdf.to_file(output_path, driver="GPKG")
                # print(f"{file_name} saved successfully.")

            except Exception as e:
                print(f"Error merging {file_name}: {e}")
        else:
            print(f"No valid data found for {file_name}, skipping save.")


from rasterio.mask import mask
from shapely.geometry import box, mapping
import os
from pathlib import Path
import rasterio

def clip_tiff_by_bbox(city, data_tiff_path, output_path,
                      min_lon, min_lat, max_lon, max_lat):
    """
    Clip all GeoTIFF files in a folder by a bounding box,
    and save clipped rasters to a new folder.

    Parameters:
    - city (str): City name used to name output folder
    - data_tiff_path (Path): Folder containing input GeoTIFF files
    - output_path (Path): Folder to save clipped GeoTIFF files
    - min_lon, min_lat, max_lon, max_lat (float): bounding box coords

    Returns:
    - output_dir (Path): Path of folder containing clipped TIFFs
    """
    if not isinstance(data_tiff_path, Path):
        data_tiff_path = Path(data_tiff_path)
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    tiff_files = sorted(data_tiff_path.glob("*.tif"))
    n_task = len(tiff_files)

    output_dir = output_path / f"{city}-NTL-clipped"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    geo = [mapping(bbox)]

    for index, tiff_path in enumerate(tiff_files):
        print(f"Processing {index + 1}/{n_task}: {tiff_path.name}")

        with rasterio.open(tiff_path) as src:
            out_image, out_transform = mask(src, geo, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        output_tiff_path = output_dir / tiff_path.name

        with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"All clipped TIFF files saved to {output_dir}")
    return output_dir

from pathlib import Path
from typing import List, Union
import geopandas as gpd
from tqdm import tqdm

def load_gpkgs(
    data_folder: Path, 
    dates: Union[List[str], None] = None,  
    date_format: str = "%Y-%m-%d"
) -> List[gpd.GeoDataFrame]:
    """
    Read all GPKG files matching given dates into a list of GeoDataFrames.

    Parameters
    ----------
    data_folder : Path
        Folder containing the .gpkg files.
    dates : list of str or None
        List of date strings to filter filenames. If None, loads all files.
    date_format : str
        Date format pattern to parse date strings if needed (currently unused, for future use).

    Usage
    -----
    >>> load_gpkgs(DATA_PATH / "addis-mesh-data", dates=['2023-05-01', '2023-05-02'])
    """
    gpkg_files = sorted(data_folder.glob('*.gpkg'))

    if dates is not None:
        
        filtered_files = []
        for f in gpkg_files:
            fname = f.stem
            if any(date_str in fname for date_str in dates):
                filtered_files.append(f)
        gpkg_files = filtered_files

    gdfs = []
    for file in tqdm(gpkg_files, desc="Reading GPKG files"):
        try:
            gdf = gpd.read_file(file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Failed to read {file}, error: {e}")
    return gdfs


def fill_tci_to_gpkg(
        gpkg_folder: Path, 
        tci_csv_folder: Path, 
        output_folder: Path
):
    """
    Inject TCI values into each mesh GPKG file and save to a new GPKG file.

    Parameters
    ----------
    - gpkg_folder: Path 
        Path to the folder containing empty mesh GPKG files.
    - tci_csv_folder: Path
        Path to the folder containing 'tci_baghdad_2023.csv' and 'tci_baghdad_2024.csv'.
    - output_folder: Path
        Path to the folder where the output GPKG files will be saved.
    """
    output_folder.mkdir(exist_ok=True, parents=True)

    # Read and merge TCI data from both years
    df1 = pd.read_csv(tci_csv_folder / 'tci_baghdad_2023.csv')
    df2 = pd.read_csv(tci_csv_folder / 'tci_baghdad_2024.csv')
    df = pd.concat([df1, df2], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    df = df.fillna(0)  # Set NA to 0

    # Complete full date range from 2023-01-01 to 2024-12-31
    non_date_cols = ["geom_id", "geometry"]
    date_cols = [col for col in df.columns if col not in non_date_cols]

    full_dates = pd.date_range(start="2023-01-01", end="2024-12-31")
    full_date_strs = [d.strftime("%Y-%m-%d") for d in full_dates]

    # Add missing date columns with NaN
    missing_dates = sorted(set(full_date_strs) - set(date_cols))
    for missing in missing_dates:
        df[missing] = 0.0

    # Reorder columns: non-date columns first, then sorted dates
    ordered_columns = non_date_cols + sorted(full_date_strs)
    df = df[ordered_columns]

    # Iterate through each GPKG file
    for gpkg_path in gpkg_folder.glob("*.gpkg"):
        try:
            layer_name = fiona.listlayers(gpkg_path)[0]
            output_name = gpkg_path.stem
            date_str = gpkg_path.stem.split("baghdad-")[-1].split(".gpkg")[0]

            print(f"Processing file: {output_name}")

            gdf = gpd.read_file(gpkg_path, layer=layer_name)

            if date_str not in df.columns:
                print(f"[Skipped] {date_str} not found in TCI data")
                continue

            # Inject TCI value for that specific date
            gdf["TCI"] = df[date_str].values  # Ensure row count matches 

            output_file = output_folder / f"{output_name}.gpkg"
            gdf.to_file(output_file, layer=layer_name, driver="GPKG")

        except Exception as e:
            print(f"[Error] Failed to process {gpkg_path.name}: {e}")



import geopandas as gpd
import pandas as pd
import fiona
from pathlib import Path

def fill_tci_to_gpkg(
        gpkg_folder: Path, 
        tci_csv_folder: Path, 
        output_folder: Path
):
    """
    Inject TCI values into each mesh GPKG file and save to a new GPKG file.

    Parameters
    ----------
    - gpkg_folder: Path 
        Path to the folder containing empty mesh GPKG files.
    - tci_csv_folder: Path
        Path to the folder containing 'tci_baghdad_2023.csv' and 'tci_baghdad_2024.csv'.
    - output_folder: Path
        Path to the folder where the output GPKG files will be saved.
    """
    output_folder.mkdir(exist_ok=True, parents=True)

    # Read and merge TCI data from both years
    df1 = pd.read_csv(tci_csv_folder / 'tci_baghdad_2023.csv')
    df2 = pd.read_csv(tci_csv_folder / 'tci_baghdad_2024.csv')
    df = pd.concat([df1, df2], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    df = df.fillna(0)  # Set NA to 0

    # Complete full date range from 2023-01-01 to 2024-12-31
    non_date_cols = ["geom_id", "geometry"]
    date_cols = [col for col in df.columns if col not in non_date_cols]

    full_dates = pd.date_range(start="2023-01-01", end="2024-12-31")
    full_date_strs = [d.strftime("%Y-%m-%d") for d in full_dates]

    # Add missing date columns with NaN
    missing_dates = sorted(set(full_date_strs) - set(date_cols))
    for missing in missing_dates:
        df[missing] = 0.0

    # Reorder columns: non-date columns first, then sorted dates
    ordered_columns = non_date_cols + sorted(full_date_strs)
    df = df[ordered_columns]

    # Iterate through each GPKG file
    for gpkg_path in gpkg_folder.glob("*.gpkg"):
        try:
            layer_name = fiona.listlayers(gpkg_path)[0]
            output_name = gpkg_path.stem
            date_str = gpkg_path.stem.split("baghdad-")[-1].split(".gpkg")[0]

            print(f"Processing file: {output_name}")

            gdf = gpd.read_file(gpkg_path, layer=layer_name)

            if date_str not in df.columns:
                print(f"[Skipped] {date_str} not found in TCI data")
                continue

            # Inject TCI value for that specific date
            gdf["TCI"] = df[date_str].values  # Ensure row count matches 

            output_file = output_folder / f"{output_name}.gpkg"
            gdf.to_file(output_file, layer=layer_name, driver="GPKG")

        except Exception as e:
            print(f"[Error] Failed to process {gpkg_path.name}: {e}")


def clip_cloud_tiff_by_bbox(city, data_tiff_path, output_path,
                    min_lon, min_lat, max_lon, max_lat):
    """
    Clip all GeoTIFF files in a folder by a bounding box,
    and save clipped rasters to a new folder.

    Parameters:
    - city (str): City name used to name output folder
    - data_tiff_path (Path): Folder containing input GeoTIFF files
    - output_path (Path): Folder to save clipped GeoTIFF files
    - min_lon, min_lat, max_lon, max_lat (float): bounding box coords

    Returns:
    - output_dir (Path): Path of folder containing clipped TIFFs
    """
    if not isinstance(data_tiff_path, Path):
        data_tiff_path = Path(data_tiff_path)
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    tiff_files = sorted(data_tiff_path.glob("*.tif"))
    n_task = len(tiff_files)

    output_dir = output_path / f"{city}-cloud-clipped"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    geo = [mapping(bbox)]

    for index, tiff_path in enumerate(tiff_files):
        print(f"Processing {index + 1}/{n_task}: {tiff_path.name}")

        with rasterio.open(tiff_path) as src:
            out_image, out_transform = mask(src, geo, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        output_tiff_path = output_dir / tiff_path.name

        with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"All clipped TIFF files saved to {output_dir}")
    return output_dir

def clip_tiff_temp_by_bbox(city, data_tiff_path, output_path,
                    min_lon, min_lat, max_lon, max_lat):
    """
    Clip all GeoTIFF files in a folder by a bounding box,
    and save clipped rasters to a new folder.

    Parameters:
    - city (str): City name used to name output folder
    - data_tiff_path (Path): Folder containing input GeoTIFF files
    - output_path (Path): Folder to save clipped GeoTIFF files
    - min_lon, min_lat, max_lon, max_lat (float): bounding box coords

    Returns:
    - output_dir (Path): Path of folder containing clipped TIFFs
    """
    if not isinstance(data_tiff_path, Path):
        data_tiff_path = Path(data_tiff_path)
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    tiff_files = sorted(data_tiff_path.glob("*.tif"))
    n_task = len(tiff_files)

    output_dir = output_path / f"{city}-temp-clipped"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    geo = [mapping(bbox)]

    for index, tiff_path in enumerate(tiff_files):
        print(f"Processing {index + 1}/{n_task}: {tiff_path.name}")

        with rasterio.open(tiff_path) as src:
            out_image, out_transform = mask(src, geo, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        output_tiff_path = output_dir / tiff_path.name

        with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"All clipped TIFF files saved to {output_dir}")
    return output_dir

def clip_tiff_temp_by_bbox(city, data_tiff_path, output_path,
                    min_lon, min_lat, max_lon, max_lat):
    """
    Clip all GeoTIFF files in a folder by a bounding box,
    and save clipped rasters to a new folder.

    Parameters:
    - city (str): City name used to name output folder
    - data_tiff_path (Path): Folder containing input GeoTIFF files
    - output_path (Path): Folder to save clipped GeoTIFF files
    - min_lon, min_lat, max_lon, max_lat (float): bounding box coords

    Returns:
    - output_dir (Path): Path of folder containing clipped TIFFs
    """
    if not isinstance(data_tiff_path, Path):
        data_tiff_path = Path(data_tiff_path)
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    tiff_files = sorted(data_tiff_path.glob("*.tif"))
    n_task = len(tiff_files)

    output_dir = output_path / f"{city}-temp-clipped"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    geo = [mapping(bbox)]

    for index, tiff_path in enumerate(tiff_files):
        print(f"Processing {index + 1}/{n_task}: {tiff_path.name}")

        with rasterio.open(tiff_path) as src:
            out_image, out_transform = mask(src, geo, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        output_tiff_path = output_dir / tiff_path.name

        with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"All clipped TIFF files saved to {output_dir}")
    return output_dir

def specific_date(start_date: str, end_date: str, time_resolution: str = 'D') -> List[str]:
    """
    Generate a list of dates within specified time period and resolution.

    Parameters:
    - start_date: str
        Start date, format: 'YYYY-MM-DD'.
    - end_date: str
        End date, format: 'YYYY-MM-DD'.
    - time_resolution: str
        Time resolution (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly). Default is 'D'.
    
    Return:
    - dates(list): List of date strings marking the ends of each time segment, format: 'YYYY-MM-DD'.
    
    """
    dates = (
        pd.date_range(start_date, end_date, freq = time_resolution)
        .strftime('%Y-%m-%d')
        .tolist()
    )
    return dates

import pandas as pd
from typing import List


# Function: generate desired time period of NO2 data  
def specific_date(start_date: str, end_date: str, time_resolution: str = 'D') -> List[str]:
    """
    Generate a list of dates within specified time period and resolution.

    Parameters:
    - start_date: str
        Start date, format: 'YYYY-MM-DD'.
    - end_date: str
        End date, format: 'YYYY-MM-DD'.
    - time_resolution: str
        Time resolution (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly). Default is 'D'.
    
    Return:
    - dates(list): List of date strings marking the ends of each time segment, format: 'YYYY-MM-DD'.
    
    """
    dates = (
        pd.date_range(start_date, end_date, freq = time_resolution)
        .strftime('%Y-%m-%d')
        .tolist()
    )
    return dates

def clip_raster_with_shapefile_vrt(input_tiff_list, shapefile, output_tiff, nodata_value=255):
    """
    Merge multiple TIFFs into a VRT and clip it using shapefile boundary
    """
    from rasterio.merge import merge
    import rasterio
    from rasterio.mask import mask
    import geopandas as gpd

    src_files_to_mosaic = [rasterio.open(str(tif)) for tif in input_tiff_list]
    mosaic, transform = merge(src_files_to_mosaic)
    meta = src_files_to_mosaic[0].meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "nodata": nodata_value
    })

    shapes = gpd.read_file(shapefile)
    geometry = [shapes.geometry.iloc[0].__geo_interface__]

    with rasterio.open(output_tiff, "w", **meta) as dst:
        dst.write(mosaic)

    with rasterio.open(output_tiff) as src:
        out_image, out_transform = mask(src, geometry, crop=True, nodata=nodata_value)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_value
        })

    with rasterio.open(output_tiff, "w", **out_meta) as dest:
        dest.write(out_image)

def revert_tiff_filenames_to_match_mesh(folder_path):
    import os
    import re

    for file in os.listdir(folder_path):
        if file.endswith("_filled.tif"):
            new_name = re.sub(r".*_(\d{4}_\d{2}_\d{2})_filled.tif", r"addis-ababa_LST_\1_filled.tif", file)
            os.rename(folder_path / file, folder_path / new_name)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

def get_esa_landcover_colormap():
    """
    Returns ESA WorldCover 2021 class values, names, colormap, and normalization object.

    Returns
    -------
    class_values : list of int
        ESA land cover class values.
    class_names : list of str
        Human-readable names for each class.
    cmap : ListedColormap
        Colormap object for visualization.
    norm : BoundaryNorm
        Normalization object matching class values to color bins.
    """

    class_values = [
        10, 20, 30, 40, 50, 60, 70, 80,
        90, 95, 100, 111, 112, 113
    ]

    class_names = [
        "Tree cover", "Shrubland", "Grassland", "Cropland",
        "Built-up", "Bare/sparse vegetation", "Snow and ice", "Water bodies",
        "Wetlands", "Mangroves", "Moss and lichen", "Permanent snow", "Glaciers", "Others"
    ]

    class_colors = [
        "#006400",  # 10 Tree cover (dark green)
        "#ffbb22",  # 20 Shrubland (light brown)
        "#ffff4c",  # 30 Grassland (yellow)
        "#f096ff",  # 40 Cropland (pink)
        "#fa0000",  # 50 Built-up (red)
        "#b4b4b4",  # 60 Sparse vegetation (gray)
        "#f0f0f0",  # 70 Snow and ice (white)
        "#0064c8",  # 80 Water bodies (blue)
        "#0096a0",  # 90 Wetlands (teal)
        "#00cf75",  # 95 Mangroves (greenish)
        "#fae6a0",  # 100 Moss and lichen (beige)
        "#dcdcdc",  # 111 Permanent snow (light gray)
        "#b0e0e6",  # 112 Glaciers (pale blue)
        "#a0a0a0",  # 113 Others/unclassified
    ]

    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(class_values + [114], cmap.N)

    return class_values, class_names, cmap, norm
