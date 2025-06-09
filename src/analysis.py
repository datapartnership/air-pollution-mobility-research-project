from collections import defaultdict
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

def compute_pacf(
        data_folder: Path, 
        nlags: int = 20, 
        column_name: str = 'no2_mean',
        pacf_title: str = "Mean PACF across all geom_ids"
):
    """
    Reads all .gpkg files in the given folder, extracts the specified field for each mesh cell over time,
    computes the PACF for each cell, and plots the average PACF curve across all cells.

    Parameters
    -----------
    data_folder: Path
        Path to the folder containing GPKG files.
    nlags: int
        Maximum number of lags for PACF.
    column_name: str 
        Name of the column to analyze (e.g., "no2_mean").

    Returns
    -------
    pacf_all: np.ndarray
        All PACF values of all mesh cells.

    Usage
    -----
    >>> data_folder = DATA_PATH / "addis-mesh-data"
    >>> all_pacf = compute_and_plot_mean_pacf(data_folder, nlags=20, column_name='no2_mean')

    """
    # Load all GPKG files
    gpkg_files = sorted(data_folder.glob('*.gpkg'))
    gdfs = []

    for file in tqdm(gpkg_files, desc="Reading GPKG files"):
        try:
            gdf = gpd.read_file(file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Failed to read {file}, error: {e}")

    # Build time series for each geom_id
    series_dict = defaultdict(list)
    for gdf in gdfs:
        for _, row in gdf.iterrows():
            geom_id = row['geom_id']
            value = row[column_name]
            series_dict[geom_id].append(value)

    # Convert to DataFrame: each column = one geom_id, each row = one day
    df = pd.DataFrame(series_dict)
    df = df.interpolate(limit_direction='both')     # Fill missing values if any

    # Compute PACF for each geom_id
    pacf_all = []
    for geom_id in tqdm(df.columns, desc="Calculating PACF for each geom_id"):
        series = df[geom_id].values
        if np.isnan(series).any():
            continue  # Skip if NaNs remain
        try:
            pacf_vals = pacf(series, nlags=nlags, method='ols')
            pacf_all.append(pacf_vals)
        except Exception as e:
            print(f"PACF failed for geom_id {geom_id}: {e}")

    if not pacf_all:
        print("No PACF values computed. All time series may have issues.")
        return None

    # Convert to numpy array and compute mean PACF across all cells
    pacf_all = np.array(pacf_all)
    return pacf_all


import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List
pd.set_option('future.no_silent_downcasting', True)

def average_mesh_over_time(
        gdfs: List[gpd.GeoDataFrame]
) -> gpd.GeoDataFrame:
    """
    Calculate the temporal average of numerical features from a list of GeoDataFrames (meshes).
    
    Parameters
    -----------
    gdfs : list of GeoDataFrame
        Each GeoDataFrame represents a mesh at a different time point. 
        All GeoDataFrames must have the same geometry and ordering.
    
    Returns
    --------
    mean_gdf : GeoDataFrame
        A GeoDataFrame representing the spatial mesh with numerical features averaged over time.
        The 'geom_id' and 'geometry' columns are taken from the first input GeoDataFrame.

    Usage
    ------
    >>> mean_mesh = average_mesh_over_time(gdfs)
    """

    # Get numeric columns, excluding 'geom_id' if present
    numeric_cols = gdfs[0].select_dtypes(include=[np.number]).columns.tolist()
    if "geom_id" in numeric_cols:
        numeric_cols.remove("geom_id")
    
    cleaned_arrays = []
    for gdf in gdfs:
        # Replace None with np.nan to handle missing values
        temp_df = gdf[numeric_cols].replace({None: np.nan}).copy()
        # Infer proper data types to avoid future warnings
        temp_df = temp_df.infer_objects(copy=False)
        cleaned_arrays.append(temp_df.values)
    
    # Convert list of arrays to 3D numpy array: (time, n_cells, n_features)
    data_matrix = np.array(cleaned_arrays)
    
    # Compute mean along time axis, ignoring NaNs
    mean_data = np.nanmean(data_matrix, axis=0)
    
    # Construct a new GeoDataFrame using geometry and geom_id from first gdf
    mean_gdf = gdfs[0][["geom_id", "geometry"]].copy()
    # Assign averaged numeric features
    mean_gdf[numeric_cols] = mean_data
    
    return mean_gdf


import warnings
from libpysal.weights import KNN
from esda import Moran_Local
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

def compute_plot_local_moran(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    feature_col: str,
    k_neighbors: int = 8,
    alpha: float = 0.05,
    plot_title: str = None,
    if_emphasize: bool = True,
    cmap: str = "hot_r",
    figsize: tuple = (10, 8),
    show_plot: bool = True
) -> gpd.GeoDataFrame:
    """
    Compute Local Moran's I statistic for a numeric feature in a GeoDataFrame and visualize it.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Input spatial data. Must contain geometry and the specified numeric feature column.
    feature_col : str
        The column name of the numeric feature to analyze.
    k_neighbors : int, optional
        Number of nearest neighbors used to construct the spatial weight matrix (default is 8).
    alpha : float, optional
        Significance level used to identify statistically significant clusters (default is 0.05).
    plot_title : str, optional
        Title of the output plot. Defaults to "Local Moran's I (feature_col)".
    cmap : str, optional
        Colormap used for plotting the Local Moran's I values (default is "hot_r").
    figsize : tuple, optional
        Size of the output figure (default is (10, 8)).
    show_plot : bool, optional
        Whether to display the plot immediately (default is True).
    
    Returns
    -------
    gdf_result : GeoDataFrame
        The original GeoDataFrame with added columns:
        - 'local_I': Local Moran's I value for each feature.
        - 'p_sim': P-value from the permutation test.
        - 'significant': Boolean indicating statistical significance (p < alpha).
    """
    
    # Construct K-Nearest Neighbors weight matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = KNN.from_dataframe(gdf, k=k_neighbors)
    w.transform = 'r'  # Row-standardized weights

    # Extract the feature values
    x = gdf[feature_col].values

    # Calculate Local Moran's I
    moran_local = Moran_Local(x, w)

    # Copy the original GeoDataFrame and append results
    gdf_result = gdf.copy()
    gdf_result["local_I"] = moran_local.Is
    gdf_result["p_sim"] = moran_local.p_sim
    gdf_result["significant"] = moran_local.p_sim < alpha

    # Set default plot title if none is given
    if plot_title is None:
        plot_title = f"Local Moran's I ({feature_col})"

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf_result.plot(column="local_I", cmap=cmap, legend=True, ax=ax)

    # Outline statistically significant areas
    if if_emphasize:
        sig = gdf_result[gdf_result["significant"]]
        if not sig.empty:
            sig.boundary.plot(ax=ax, color='black', linewidth=1)

    ax.set_title(plot_title)
    ax.set_axis_off()
    plt.savefig(output_path / f'{plot_title}.png', dpi=300)
    print(f"Figure saved to {output_path}")

    if show_plot:
        plt.show()

    # return gdf_result
