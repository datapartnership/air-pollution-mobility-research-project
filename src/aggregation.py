import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import fiona

# Function: Aggregate data within each mesh
def aggregate_to_mesh(mesh_grid, tiff_path:Path):
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
    mesh_grid["no2_mean"] = [s["mean"] for s in stats]

    return mesh_grid

# Function: Aggregate and write aggregated values to multiple meshes
def aggregate_data(
        data_tiff_path:Path, 
        mesh_path: Path,
        layer_name: str
):
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
            new_mesh = aggregate_to_mesh(mesh, tiff_path)
            new_mesh.to_file(matched_mesh, driver='GPKG', layer=layer_name, mode='w')


        except Exception as e:
            print(f"Error processing {tiff_path} at index {index}: {e}")
            print("Skipping to next file...")
            continue