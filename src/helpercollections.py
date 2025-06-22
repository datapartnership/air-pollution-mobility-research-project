import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
from pathlib import Path
import os

from tqdm import tqdm
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
    >>> DATA_PATH = Path("your/data/root/path")

    >>> feature_mesh_paths = [DATA_PATH / "addis-no2-mesh-data", DATA_PATH / "addis-OSM-mesh-data", DATA_PATH / "addis-pop-mesh-data"]

    >>> output_folder = DATA_PATH / "addis-mesh-data"

    >>> output_folder.mkdir(exist_ok=True)
    
    >>> merge_multiple_gpkgs(feature_mesh_paths, output_folder)
        
    """

    # Make sure the output folder exists
    output_folder.mkdir(exist_ok=True)
    
    # Get list of file names from the first directory (assume same files in all)
    file_names = [f.name for f in feature_mesh_paths[0].glob("*.gpkg")]

    for file_name in tqdm(file_names, desc="Progress", total=len(file_names)):
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

import os
from pathlib import Path
import re

def revert_tiff_filenames_to_match_mesh(tiff_folder: Path) -> None:
    """
    Rename filled TIFF files to match mesh naming convention: city-YYYY-MM-DD.tif

    Example:
    From: addis-ababa_LST_2023-01-01_filled.tif
    To:   addis-ababa-2023-01-01.tif

    Parameters:
    - tiff_folder (Path): folder containing filled TIFF files
    """
    if not isinstance(tiff_folder, Path):
        tiff_folder = Path(tiff_folder)

    tif_files = list(tiff_folder.glob("*_filled.tif"))
    renamed = 0

    for tif in tif_files:
        match = re.search(r"(.+)_LST_(\d{4}-\d{2}-\d{2})_filled\.tif", tif.name)
        if match:
            city = match.group(1)
            date_str = match.group(2)
            new_name = f"{city}-{date_str}.tif"
            new_path = tif.parent / new_name
            tif.rename(new_path)
            renamed += 1
        else:
            print(f" Cannot extract date from {tif.name}, skipping.")

    print(f"Renamed {renamed} TIFF files to standard format.")


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

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping

def clip_raster_with_shapefile(input_tiff, shapefile, output_tiff, nodata_value=255):
    """
    Clip a raster TIFF file using the geometry of a shapefile.

    Parameters
    ----------
    input_tiff : Path
        Path to the input raster TIFF file.
    shapefile : Path
        Path to the shapefile used for clipping.
    output_tiff : Path
        Path to save the clipped raster.
    nodata_value : int or float, optional
        Value to assign to nodata areas in the output raster. Default is 255.
    """
    # Load shapefile geometry
    shapes = gpd.read_file(shapefile)
    geometry = [mapping(shapes.geometry.iloc[0])]

    with rasterio.open(input_tiff) as src:
        out_image, out_transform = mask(src, geometry, crop=True, nodata=nodata_value)
        out_meta = src.meta.copy()

    # Update metadata for the output raster
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": nodata_value
    })

    # Save clipped raster
    with rasterio.open(output_tiff, "w", **out_meta) as dest:
        dest.write(out_image)


from rasterio.mask import mask
from shapely.geometry import box, mapping
from pathlib import Path
import rasterio
import os

def clip_tiff_with_bbox(city, data_tiff_path, output_path,
                        min_lon, min_lat, max_lon, max_lat):
    """
    Clip all GeoTIFF files in a folder using a bounding box and save them 
    into a structured output folder specific to LST.

    Parameters:
    - city (str): city name (used in naming output folder)
    - data_tiff_path (Path): folder containing input GeoTIFFs
    - output_path (Path): folder to save clipped GeoTIFFs
    - min_lon, min_lat, max_lon, max_lat (float): bounding box coordinates

    Returns:
    - Path: folder containing all clipped GeoTIFFs
    """
    if not isinstance(data_tiff_path, Path):
        data_tiff_path = Path(data_tiff_path)
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    tiff_files = sorted(data_tiff_path.glob("*.tif"))
    if not tiff_files:
        print(f"⚠️ No TIFF files found in {data_tiff_path}")
        return None

    output_dir = output_path / f"{city}-LST-clipped"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    geo = [mapping(bbox)]

    for idx, tif in enumerate(tiff_files, 1):
        print(f"📦 Clipping {idx}/{len(tiff_files)}: {tif.name}")
        try:
            with rasterio.open(tif) as src:
                out_image, out_transform = mask(src, geo, crop=True)
                out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            out_tif_path = output_dir / tif.name
            with rasterio.open(out_tif_path, "w", **out_meta) as dest:
                dest.write(out_image)
        except Exception as e:
            print(f"❌ Failed to clip {tif.name}: {e}")

    print(f"✅ All clipped TIFF files saved to {output_dir}")
    return output_dir


import geopandas as gpd
def gpkg_2_shp(gpkg_root, shp_name):
    """
    Change gpkg file to shp file to assist ArcGIS visualization
    """
    gdf = gpd.read_file(gpkg_root)
    gdf.to_file(gpkg_root / shp_name, driver="ESRI Shapefile")



import re
from tqdm import tqdm
# Function: Aggregate and write aggregated values to multiple meshes
def write_single_column_to_new_mesh(
        folder_path:Path, 
        empty_mesh_path: Path,
        old_name: str,
        new_name: str,
        old_layer_name: str, 
        layer_name: str)-> None :
    """
    Iterate through all GPKG files in the specified folder, updating mesh files with data columns extracted from time-series GPKG files.

    1. Extract the date from the filename.
    2. Find the corresponding empty mesh file in another folder that contains the same date in its filename.
    3. Read the specified layer from the original GPKG file and extract the column named `old_name`.
    4. Assign the extracted column values to the `new_name` column of the matched empty mesh file.
    5. Save the updated mesh file with the specified `layer_name`.

    Parameters
    ----------
    folder_path : Path
        Path to the folder containing source GPKG files. Filenames must include a date string in the format 'YYYY-MM-DD', e.g. 'addis-ababa-2023-01-02.gpkg'.
    empty_mesh_path : Path
        Path to the folder containing empty mesh GPKG files. Filenames should also contain date strings to allow matching.
    old_name : str
        The column name to extract from the source GPKG files.
    new_name : str
        The column name to assign in the empty mesh files.
    old_layer_name : str
        The layer name to read from the source GPKG files, must be known to make sure successfully load the layer.
    layer_name : str
        The layer name to write in the new mesh files.

    Returns
    -------
    None

    Example
    -----
    >>> folder_path = DATA_PATH / '1'
    >>> empty_mesh_path = DATA_PATH / 'addis-temp-mesh-data'
    >>> old_name = 'LST_day_mean'
    >>> new_name = 'temp_mean'
    >>> old_layer_name = 'LST_day'


    >>> write_single_column_to_new_mesh(
                        folder_path=folder_path, 
                        empty_mesh_path=empty_mesh_path, 
                        old_name=old_name, 
                        new_name=new_name, 
                        old_layer_name=old_layer_name, 
                        layer_name = lyr_addis_name)
    """

    
    gpkg_files = [f for f in folder_path.glob("*gpkg")]
    abs_file_paths = [folder_path / f for f in gpkg_files] # absolute path
    n_task = len(gpkg_files)
    
    for index, abs_path in tqdm(enumerate(abs_file_paths), desc="Processing GPKG files: ", total = n_task):

        filename = abs_path.name
        match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
        if match:
            date = match.group()
        else:
            print(f"Warning: can't find date in filename {filename}")
            continue

        try:
            old_mesh = gpd.read_file(abs_path, layer = old_layer_name)
            matched_mesh =  [f for f in empty_mesh_path.glob("*.gpkg") if date in f.name][0]  # typically only one match， absolute path
            
            new_mesh = gpd.read_file(matched_mesh)
            new_mesh[new_name] = old_mesh[old_name]
            new_mesh.to_file(matched_mesh, driver='GPKG', layer=layer_name, mode='w')


        except Exception as e:
            print(f"Error processing {abs_path} at index {index}: {e}")
            print("Skipping to next file...")
            continue



from pathlib import Path
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

def convert_gpkgs_to_parquet(mesh_folder: Path, output_path: Path, file_name: str) -> None:
    """
    Load multiple GeoPackage files from a folder, assign sequential dates starting from 2023-01-01,
    concatenate them into a single GeoDataFrame, and save the result as a parquet file.

    Parameters:
        mesh_folder (Path): Path to the folder containing GeoPackage (.gpkg) files.
        output_path (Path): Directory path where the parquet file will be saved.
        file_name (str): Name of the output parquet file (without extension).

    Returns:
        None

    Side Effects:
        Saves the concatenated GeoDataFrame to '{output_path}/{file_name}.parquet' 
        using the PyArrow engine with Snappy compression.
    """

    file_paths = sorted(mesh_folder.glob("*.gpkg"))
    all_data = []

    for i, path in tqdm(enumerate(file_paths), total=len(file_paths)):
        date = datetime(2023, 1, 1) + timedelta(days=i)
        gdf = gpd.read_file(path)
        gdf["date"] = date
        all_data.append(gdf)

    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_parquet(output_path / f"{file_name}.parquet", engine="pyarrow", compression="snappy")


import json
from pathlib import Path

def extract_headings(ipynb_path):
    """
    Extract markdown headings from a Jupyter Notebook file.

    Args:
        ipynb_path (str or Path): Path to the input .ipynb file.

    Returns:
        list of tuples: Each tuple contains (heading_level, heading_title).
    """
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    toc = []
    for cell in data['cells']:
        if cell['cell_type'] == 'markdown':
            for line in cell['source']:
                stripped = line.lstrip()
                if stripped.startswith('#'):
                    level = len(stripped) - len(stripped.lstrip('# '))
                    title = stripped.strip('#').strip()
                    toc.append((level, title))
    return toc

# def generate_tree_lines(toc):
#     """
#     Generate a list of strings representing a tree structure
#     based on extracted headings.

#     Args:
#         toc (list of tuples): List of (level, title) tuples.

#     Returns:
#         list of str: Lines of the tree structure with ASCII/Unicode branches.
#     """
#     stack = []
#     n = len(toc)
#     lines = []

#     for i, (level, title) in enumerate(toc):
#         next_same_or_less_index = None
#         for j in range(i+1, n):
#             if toc[j][0] <= level:
#                 next_same_or_less_index = j
#                 break
#         is_last = (next_same_or_less_index is None or toc[next_same_or_less_index][0] < level)

#         while len(stack) >= level:
#             stack.pop()

#         stack.append(not is_last)

#         indent = ""
#         for idx in range(level - 1):
#             if stack[idx]:
#                 indent += "│   "
#             else:
#                 indent += "    "

#         branch = "├── " if stack[-1] else "└── "
#         lines.append(f"{indent}{branch}{title}")

#     return lines
def generate_tree_lines(toc):
    """
    Generate a list of strings representing a tree structure
    based on extracted headings.

    Args:
        toc (list of tuples): List of (level, title) tuples.

    Returns:
        list of str: Lines of the tree structure with ASCII/Unicode branches.
    """
    stack = []
    lines = []
    n = len(toc)

    for i, (level, title) in enumerate(toc):
        # Find if this is the last node at the current level
        next_same_or_less_index = None
        for j in range(i + 1, n):
            if toc[j][0] <= level:
                next_same_or_less_index = j
                break
        is_last = (next_same_or_less_index is None or toc[next_same_or_less_index][0] < level)

        # Ensure stack has exactly level-1 elements before appending
        while len(stack) > level - 1:
            stack.pop()
        while len(stack) < level - 1:
            stack.append(True)  # filler to avoid index error

        stack.append(not is_last)

        indent = ""
        for idx in range(level - 1):
            if stack[idx]:
                indent += "│   "
            else:
                indent += "    "

        branch = "├── " if not is_last else "└── "
        lines.append(f"{indent}{branch}{title}")

    return lines


def export_notebook_toc(ipynb_path, output_path):
    """
    Extract the table of contents from a Jupyter Notebook and save
    it as a tree-structured text file.

    Args:
        ipynb_path (str or Path): Path to the input notebook (.ipynb).
        output_path (str or Path): Path to the output text file.
    """
    name = ipynb_path.stem
    toc = extract_headings(ipynb_path)
    lines = generate_tree_lines(toc)
    for line in lines:
        print(line) 
    with open(output_path / f'{name}_TOC.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved Table of Content to: {output_path}")
