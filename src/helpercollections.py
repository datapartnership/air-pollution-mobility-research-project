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
from tqdm import tqdm
from typing import List

def load_gpkgs(data_folder: Path) -> List[gpd.GeoDataFrame]:
    """"
    Read all the gpkg into one list, each element being one mesh, GeoDataFrame.

    Usage:
    >>> load_gpkgs(DATA_PATH / "addis-mesh-data")
    """
    gpkg_files = sorted(data_folder.glob('*.gpkg'))
    gdfs = []
    for file in tqdm(gpkg_files, desc="Reading GPKG files"):
        try:
            gdf = gpd.read_file(file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Failed to read {file}, error: {e}")
    return gdfs


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

