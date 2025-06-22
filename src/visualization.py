# This script defines the functions that are used to generate related visualization,
# including:
#   (1) static plots: plot_percent_clipped_raster()
#   (2) dynamic plots (animation): tiff_2_gif(), mesh_2_gif()


import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import shutil
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from typing import List, Union, Tuple, Optional
from pathlib import Path


def plot_raster(
    arr: np.ndarray,
    percent_clip: float = 0.5,
    title: str = "Raster Visualization",
    colors: Union[List[str], None] = None,
    ax: Optional[plt.Axes] = None,
    cbar: bool = True,
    cbar_label: str = "NO₂ (percent-clipped)",
    imshow_kwargs: Optional[dict] = None,
    cbar_kwargs: Optional[dict] = None,
) -> Tuple[plt.Axes, Optional[plt.Axes]]:
    """
    Display a 2D raster image with percent-based value clipping and custom colormap.

    Parameters
    ----------
    arr : np.ndarray
        2D raster array with missing values marked as np.nan.
    percent_clip : float, default 0.5
        Percentage of the lower and upper tails to clip when computing color bounds.
    colors : list[str] | None, optional
        List of color names to define a custom colormap.
    ax : matplotlib.axes.Axes | None, optional
        Existing matplotlib axes to draw on. If None, a new figure and axes are created.
    cbar : bool, default True
        Whether to include a colorbar in the plot.
    cbar_label : str, default "NO₂ (percent-clipped)"
        Label for the colorbar.
    imshow_kwargs : dict | None, optional
        Additional keyword arguments to pass to `imshow()`.
    cbar_kwargs : dict | None, optional
        Additional keyword arguments to pass to `colorbar()`.

    Returns
    -------
    ax : plt.Axes
        The axes containing the image.
    cbar_ax : plt.Axes or None
        The colorbar axes if `cbar` is True; otherwise None.
    """
    if colors is None:
        colors = ["blue", "cyan", "yellow", "red"]

    # Flatten and clean the array to compute percentiles
    flat = arr[~np.isnan(arr)].ravel()
    vmin = np.percentile(flat, percent_clip)
    vmax = np.percentile(flat, 100 - percent_clip)

    # Normalize and create colormap
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = LinearSegmentedColormap.from_list("custom_map", colors)

    # Create new figure and axes if none provided
    created_new_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        created_new_ax = True

    # Apply imshow with extra kwargs
    imshow_kwargs = imshow_kwargs or {}
    img = ax.imshow(arr, cmap=cmap, norm=norm, **imshow_kwargs)
    ax.set_axis_off()
    ax.set_title(title)

    # Add colorbar if requested
    cbar_ax = None
    if cbar:
        cbar_kwargs = cbar_kwargs or {}
        cbar_ax = plt.colorbar(img, ax=ax, label=cbar_label, **cbar_kwargs)

    if created_new_ax:
        plt.show()

    return ax, cbar_ax



def plot_raster_ntl(
    arr: np.ndarray,
    percent_clip: float = 0.5,
    title: str = "Raster Visualization",
    colors: Union[List[str], None] = None,
    ax: Optional[plt.Axes] = None,
    cbar: bool = True,
    cbar_label: str = "Night Time Light (percent-clipped)",
    imshow_kwargs: Optional[dict] = None,
    cbar_kwargs: Optional[dict] = None,
) -> Tuple[plt.Axes, Optional[plt.Axes]]:
    """
    Display a 2D raster image with percent-based value clipping and custom colormap.

    Parameters
    ----------
    arr : np.ndarray
        2D raster array with missing values marked as np.nan.
    percent_clip : float, default 0.5
        Percentage of the lower and upper tails to clip when computing color bounds.
    colors : list[str] | None, optional
        List of color names to define a custom colormap.
    ax : matplotlib.axes.Axes | None, optional
        Existing matplotlib axes to draw on. If None, a new figure and axes are created.
    cbar : bool, default True
        Whether to include a colorbar in the plot.
    cbar_label : str, default "NO₂ (percent-clipped)"
        Label for the colorbar.
    imshow_kwargs : dict | None, optional
        Additional keyword arguments to pass to `imshow()`.
    cbar_kwargs : dict | None, optional
        Additional keyword arguments to pass to `colorbar()`.

    Returns
    -------
    ax : plt.Axes
        The axes containing the image.
    cbar_ax : plt.Axes or None
        The colorbar axes if `cbar` is True; otherwise None.
    """
    if colors is None:
        colors = ["blue", "cyan", "yellow", "red"]

    # Flatten and clean the array to compute percentiles
    flat = arr[~np.isnan(arr)].ravel()
    vmin = np.percentile(flat, percent_clip)
    vmax = np.percentile(flat, 100 - percent_clip)

    # Normalize and create colormap
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = LinearSegmentedColormap.from_list("custom_map", colors)

    # Create new figure and axes if none provided
    created_new_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        created_new_ax = True

    # Apply imshow with extra kwargs
    imshow_kwargs = imshow_kwargs or {}
    img = ax.imshow(arr, cmap=cmap, norm=norm, **imshow_kwargs)
    ax.set_axis_off()
    ax.set_title(title)

    # Add colorbar if requested
    cbar_ax = None
    if cbar:
        cbar_kwargs = cbar_kwargs or {}
        cbar_ax = plt.colorbar(img, ax=ax, label=cbar_label, **cbar_kwargs)

    if created_new_ax:
        plt.show()

    return ax, cbar_ax

def plot_raster_temp(
    arr: np.ndarray,
    percent_clip: float = 0.5,
    title: str = "Raster Visualization",
    colors: Union[List[str], None] = None,
    ax: Optional[plt.Axes] = None,
    cbar: bool = True,
    cbar_label: str = "Temperature (percent-clipped)",
    imshow_kwargs: Optional[dict] = None,
    cbar_kwargs: Optional[dict] = None,
) -> Tuple[plt.Axes, Optional[plt.Axes]]:
    """
    Display a 2D raster image with percent-based value clipping and custom colormap.

    Parameters
    ----------
    arr : np.ndarray
        2D raster array with missing values marked as np.nan.
    percent_clip : float, default 0.5
        Percentage of the lower and upper tails to clip when computing color bounds.
    colors : list[str] | None, optional
        List of color names to define a custom colormap.
    ax : matplotlib.axes.Axes | None, optional
        Existing matplotlib axes to draw on. If None, a new figure and axes are created.
    cbar : bool, default True
        Whether to include a colorbar in the plot.
    cbar_label : str, default "NO₂ (percent-clipped)"
        Label for the colorbar.
    imshow_kwargs : dict | None, optional
        Additional keyword arguments to pass to `imshow()`.
    cbar_kwargs : dict | None, optional
        Additional keyword arguments to pass to `colorbar()`.

    Returns
    -------
    ax : plt.Axes
        The axes containing the image.
    cbar_ax : plt.Axes or None
        The colorbar axes if `cbar` is True; otherwise None.
    """
    if colors is None:
        colors = ["blue", "cyan", "yellow", "red"]

    # Flatten and clean the array to compute percentiles
    flat = arr[~np.isnan(arr)].ravel()
    vmin = np.percentile(flat, percent_clip)
    vmax = np.percentile(flat, 100 - percent_clip)

    # Normalize and create colormap
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = LinearSegmentedColormap.from_list("custom_map", colors)

    # Create new figure and axes if none provided
    created_new_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        created_new_ax = True

    # Apply imshow with extra kwargs
    imshow_kwargs = imshow_kwargs or {}
    img = ax.imshow(arr, cmap=cmap, norm=norm, **imshow_kwargs)
    ax.set_axis_off()
    ax.set_title(title)

    # Add colorbar if requested
    cbar_ax = None
    if cbar:
        cbar_kwargs = cbar_kwargs or {}
        cbar_ax = plt.colorbar(img, ax=ax, label=cbar_label, **cbar_kwargs)

    if created_new_ax:
        plt.show()

    return ax, cbar_ax

def plot_mesh(
    mesh: gpd.GeoDataFrame,
    feature: str,
    title: str = "Mesh Visualization",
    ax: Optional[plt.Axes] = None,
    axis_off: bool = True,  
    show_edges: bool = True,
    edgecolor: str = "grey",
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot a GeoDataFrame mesh using a specified feature column.

    Parameters
    ----------
    mesh : gpd.GeoDataFrame
        The input mesh to plot.
    feature : str
        The column in the GeoDataFrame used for coloring the polygons.
    title : str, default "Mesh Visualization"
        Title to display above the plot.
    ax : matplotlib.axes.Axes or None, optional
        Axes object to plot on. If None, a new figure and axes are created.
    axis_off : bool, default True
        Whether to hide the plot axis.
    figsize : tuple, default (8, 6)
        Size of the figure (width, height).
    show : bool, default True
        Whether to display the figure with plt.show(). 
        If plotting multiple axes, set show=False to make sure everyone can be displayed.
    **plot_kwargs :
        Additional keyword arguments passed to `GeoDataFrame.plot()`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    mesh.plot(column=feature, 
              cmap="viridis",
              edgecolor=edgecolor if show_edges else None,
              legend=True, 
              ax=ax, **plot_kwargs)
    ax.set_title(title)
    
    if axis_off:
        ax.set_axis_off()
    
    if show:
        plt.show()

    return ax


def tiff_2_gif(
        tiff_path: Path,
        output_path: Path,
        output_name: str = "timeseries",
        fps: int = 10,
        colors: Union[List[str], None] = None,
        percent_clip: float = 0.5,
):
    """
    Read all .tif files from a directory, visualise them as a time series, and export a GIF animation.

    Parameters
    ----------
    tiff_path : Path
        Directory that contains the .tif files.
    output_path: Path
        Directory that will save the output files.
    output_name : str, default "timeseries"
        Name of the output GIF file (without the extension).
    fps : int, default 12
        Frames per second for the resulting GIF.
    colors : list[str] | None, optional
        Custom colourmap expressed as a list of colour names. If *None*, the
        default is ["blue", "cyan", "yellow", "red"].
    percent_clip : float, default 0.5
        Percentage of data to clip at both tails when computing colour scale
        bounds (e.g. 0.5 clips the bottom 0.5% and the top 0.5%).

    Raises
    ------
    FileNotFoundError
        If the directory does not contain any .tif files.
    """

    # ------------------------------------------------------------
    # Collect files
    # ------------------------------------------------------------
    tif_files = sorted(tiff_path.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError("No .tif files found in the provided directory.")

    dates = [f.stem for f in tif_files]  # label frames with file stems

    # ------------------------------------------------------------
    # Determine global vmin/vmax across all rasters
    # ------------------------------------------------------------
    vals: list[np.ndarray] = []
    for tif in tqdm(tif_files, desc="Scanning percentiles"):
        with rasterio.open(tif) as src:
            arr = src.read(1)
            if src.nodata is not None:
                arr = arr[arr != src.nodata]
            arr = arr[~np.isnan(arr)]
            vals.append(arr)

    vals_flat = np.hstack(vals)
    vmin = np.percentile(vals_flat, percent_clip)
    vmax = np.percentile(vals_flat, 100 - percent_clip)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # ------------------------------------------------------------
    # Prepare colourmap
    # ------------------------------------------------------------
    if colors is None:
        colors = ["blue", "cyan", "yellow", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_map", colors)

    # ------------------------------------------------------------
    # Update function for each frame
    # ------------------------------------------------------------
    title_fmt = "Date: {date}"

    def update(i: int):
        with rasterio.open(tif_files[i]) as src:
            arr = src.read(1, masked=True)
        img.set_data(arr)
        txt.set_text(title_fmt.format(date=dates[i]))
        return img, txt

    # ------------------------------------------------------------
    # Plot first frame and set up animation
    # ------------------------------------------------------------
    with rasterio.open(tif_files[0]) as src:
        first = src.read(1, masked=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(first, cmap=cmap, norm=norm)
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04,
                 label="NO₂ (percent‑clipped)")
    txt = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        color="white", fontsize=11, ha="left", va="top",
        bbox=dict(facecolor="black", alpha=0.3, pad=2, lw=0),
    )
    ax.set_axis_off()

    anim = FuncAnimation(
        fig, update, frames=len(tif_files),
        interval=1000 / fps, blit=True,
    )

    # ------------------------------------------------------------
    # Save GIF
    # ------------------------------------------------------------
    output_dir = output_path / "animation-output"
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"{output_name}.gif"

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.show()

    print(f"Animation saved to: {gif_path}")


import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd

def mesh_2_gif(
        gpkg_path: Path,
        output_path: Path,
        output_name: str = "timeseries",
        feature: str = 'geom_id',
        fps: int = 12,
        percent_clip: float = 0.5,
):
    """
    Create a time-series GIF from a folder of GeoPackage (*.gpkg) files.

    The function assumes each GPKG file represents one time step of the same
    mesh (same CRS, similar extent).  It renders the mesh with a single
    numeric attribute (``feature``) and writes the frames to an animated GIF.

    Parameters
    ----------
    gpkg_path : pathlib.Path
        Directory that contains the *.gpkg files (one per time step).
    output_path : pathlib.Path
        Directory in which the GIF will be written.  
        A sub-folder called ``animation-output`` is created automatically.
    output_name : str, default ``"timeseries"``
        Base file name of the GIF (``<output_name>.gif``).
    feature : str, default ``"geom_id"``
        Name of the numeric column to use for colouring the mesh.
    fps : int, default ``12``
        Frames per second in the resulting GIF.
    percent_clip : float, default ``0.5``
        Percentage (0–100) to clip from both tails of the value distribution  
        when computing the colour scale.  
        *Example:* ``0.5`` → vmin = 0.5-th percentile, vmax = 99.5-th percentile.

    Workflow
    --------
    1. Scan all files and collect valid numeric values from *feature* to
       establish a global colour range (``vmin`` / ``vmax``).
    2. Build a Matplotlib ``Normalize`` object with optional percentile-based
       clipping (avoids the influence of outliers).
    3. Define an *update* callback that:
        * reads one GPKG,
        * drops invalid/empty geometries,
        * converts *feature* to numeric (non-numeric → NaN),
        * clears previous artists and replots the mesh,
        * updates an on-screen timestamp taken from ``Path.stem``.
    4. Instantiate ``FuncAnimation`` with blitting on for performance.
    5. Save the animation with a ``PillowWriter`` at *fps*.

    Raises
    ------
    FileNotFoundError
        If no ``*.gpkg`` files are found inside *gpkg_path*.
    ValueError
        * *feature* is missing from any file, **or**
        * the column holds no valid numeric data across all files.

    Notes
    -----
    * All files must share the same CRS; otherwise the overlay will shift.
    * Frames with no valid geometry or no numeric values are skipped and only
      the timestamp text is updated.
    * The function calls ``plt.show()`` for convenience.  Remove it if you do
      not want the window to pop up in non-interactive workflows.
    """
    
    # ------------------------------------------------------------
    # Collect files
    # ------------------------------------------------------------
    gpkg_files = sorted(gpkg_path.glob("*.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError("No .gpkg files found in the provided directory.")

    # ------------------------------------------------------------
    # Determine global vmin/vmax across all rasters
    # ------------------------------------------------------------
    numeric_vals = []
    for f in tqdm(gpkg_files, desc="Scanning percentiles"):
        gdf = gpd.read_file(f)

        if feature not in gdf.columns:
            # print(f"Warning: '{feature}' not found in {f.name}, skipping.")
            continue
        
        # Convert to numeric, coercing errors to NaN, then drop NaNs
        vals = pd.to_numeric(gdf[feature], errors='coerce').dropna().values
        numeric_vals.extend(vals)
    
    if len(numeric_vals) == 0:
        raise ValueError(f"No valid numeric data found in feature '{feature}' across all files.")

    vmin = np.percentile(numeric_vals, percent_clip)
    vmax = np.percentile(numeric_vals, 100 - percent_clip)
    norm = Normalize(vmin, vmax, clip=True)

    # ------------------------------------------------------------
    # Update function for each frame
    # ------------------------------------------------------------
    def update(i):
        try:
            gdf = gpd.read_file(gpkg_files[i])
        except Exception as e:
            print(f"Error reading file {gpkg_files[i]}: {e}")
            txt.set_text(f"Date: {gpkg_files[i].stem} (read error)")
            return [txt]
        
        gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)]

        if feature not in gdf.columns:
            txt.set_text(f"Date: {gpkg_files[i].stem} (missing feature)")
            return [txt]

        gdf[feature] = pd.to_numeric(gdf[feature], errors='coerce')

        ax.collections.clear()
        ax.patches.clear()

        if gdf.empty or gdf[feature].dropna().empty:
            txt.set_text(f"Date: {gpkg_files[i].stem} (no valid data)")
            return [txt]

        coll = gdf.plot(
            column=feature, ax=ax, norm=norm,
            linewidth=0, legend=False
        )
        txt.set_text(f"Date: {gpkg_files[i].stem}")
        return [*coll.collections, txt]
    
    # ------------------------------------------------------------
    # Initialize plotting
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    first_gdf = gpd.read_file(gpkg_files[0])
    # Preprocess first frame data
    first_gdf = first_gdf[first_gdf.geometry.notnull() & (~first_gdf.geometry.is_empty)]
    first_gdf[feature] = pd.to_numeric(first_gdf[feature], errors='coerce')

    coll = first_gdf.plot(
        column=feature, ax=ax, norm=norm,
        linewidth=0, legend=False
    )
    ax.set_axis_off()

    txt = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        color="white", fontsize=11, ha="left", va="top",
        bbox=dict(facecolor="black", alpha=0.3, pad=2, lw=0)
    )

    anim = FuncAnimation(
        fig, update, frames=len(gpkg_files),
        interval=1000 / fps, blit=True
    )

    # ------------------------------------------------------------
    # Save GIF
    # ------------------------------------------------------------
    output_dir = output_path / "animation-output"
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"{output_name}.gif"

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.show()

    print(f"Animation saved to: {gif_path}")


# Function to generate daily files
def generate_daily_files(src_file, year, num_days, name, output_path):
    """
    Generate daily GeoPackage files for a given year.

    Parameters:
    -----------
    src_file: Path or str
        The path of the source file to be copied.
    year: int
        The year for which daily files will be generated (used to determine start date).
    num_days: int
        Total number of daily files to create (e.g., 365 or 366).
    name: str
        The prefix name to be used in each generated file name.
    output_path: Path
        The directory where the daily files will be saved.

    Output:
    -------
    GeoPackage files (one for each day) will be created in the output directory with
    filenames formatted as `{name}-YYYY-MM-DD.gpkg`.
    """
    start_date = datetime(year, 1, 1)
    output_dir = output_path / f"pop-files-{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_days):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        filename = f"{name}-{date_str}.gpkg"
        dst_file = output_dir / filename
        shutil.copy(src_file, dst_file)

    print(f"Done: {num_days} files created for {year}.")


import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from pandas import DataFrame

def plot_corr_matrix(
        df: DataFrame, 
        cols_of_interest: List[str], 
        output_path: Path, 
        plot_name: str,
): 
    """
    Generate and save a Pearson correlation heatmap for selected features.

    This function takes a DataFrame, selects the specified columns of interest, computes 
    the Pearson correlation matrix, and saves a heatmap visualization to the given path.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the features.
    cols_of_interest : List[str]
        List of column names to include in the correlation analysis.
    output_path : Path
        Path to the directory where the heatmap image will be saved.
    plot_name : str
        Name of the output heatmap image file (without file extension).

    Returns
    -------
    None

    Example
    -------
    >>> from pathlib import Path
    >>> plot_corr_matrix(df, cols_of_interest, Path("./outputs"), "addis_corr_heatmap")
    """
    # Select relevant columns and drop rows with missing values
    df_subset = df[cols_of_interest].dropna()

    # Compute Pearson correlation matrix
    corr_matrix = df_subset.corr()

    # Plot and save heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(plot_name, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / f'{plot_name}.png', dpi=300)
    plt.show()

    print(f"Heatmap saved to {output_path / f'{plot_name}.png'}")


import matplotlib.pyplot as plt
import numpy as np

def plot_mean_pacf(
        pacf_all: np.ndarray, 
        output_path: Path,
        pacf_title: str = "Average PACF of NO$_2$ by Cell in Addis Ababa", 
        ylabel_name: str = "NO2 Concentration",
):
    """
    Plot the mean Partial Autocorrelation Function (PACF) across all mesh cells,
    including a ±1 standard deviation confidence band.

    Parameters
    ----------
    pacf_all : np.ndarray
        A 2D array of shape (n_cells, n_lags + 1), where each row contains the PACF values
        for a mesh cell at different time lags.
    output_path : Path
        Directory where the heatmap image will be saved.
    pacf_title : str, optional
        Title of the plot. Default is "Average PACF of NO$_2$ by Cell in Addis Ababa".
    ylabel_name : str, optional
        Name of the column/field analysed, used for labeling the axes. Default is "NO2 Concentration".
    
    Returns
    -------
    None

    Example
    -------
    >>> pacf_all = np.array([[1.0, 0.3, 0.1], [1.0, 0.25, 0.05]])
    >>> plot_mean_pacf(pacf_all, output_path = 'file/output', pacf_title="PACF Example", ylabel_name="NO2 Concentration")
    """
    
    # Compute the mean and standard deviation of PACF across all cells for each lag
    mean_pacf = np.mean(pacf_all, axis=0)
    pacf_std = np.std(pacf_all, axis=0)
    lags = np.arange(len(mean_pacf))

    # Use seaborn style for better aesthetics
    plt.style.use("seaborn-v0_8-darkgrid")
    # plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot stem plot for mean PACF
    markerline, stemlines, baseline = ax.stem(lags, mean_pacf, basefmt=" ")
    plt.setp(markerline, marker='o', markersize=6, color='tab:blue', label="Mean PACF")
    plt.setp(stemlines, linestyle='-', linewidth=1.5, color='tab:blue')

    # Add ±1 standard deviation confidence band
    ax.fill_between(
        lags,
        mean_pacf - pacf_std,
        mean_pacf + pacf_std,
        color='tab:blue',
        alpha=0.2,
        label='±1 Std Dev'
    )

    # Axis labels and title
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel(f"Mean PACF ({ylabel_name})", fontsize=12)
    ax.set_title(pacf_title, fontsize=14, weight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    # Add legend and adjust layout
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_path / f'{pacf_title}.png', dpi=300)
    plt.show()
    print(f"Figure saved to {output_path}")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def show_image(file_path):
    img = mpimg.imread(file_path)
    plt.figure(figsize=(10, 6)) 
    plt.imshow(img)
    plt.axis('off') 
    plt.show()
    
    
from pathlib import Path
from typing import Union, List
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm


def tiff_2_gif_cloud(
    tiff_path: Path,
    output_path: Path,
    output_name: str = "cloud_states",
    fps: int = 4,
    colors: Union[List[str], None] = None,
    class_labels: Union[List[str], None] = None,
):
    """
    Create an animated GIF from classified TIFF files (e.g., cloud state classes 0-3).

    Parameters
    ----------
    tiff_path : Path
        Directory containing input .tif files.
    output_path : Path
        Directory where the output will be saved.
    output_name : str
        Name of the output GIF file (without extension).
    fps : int
        Frames per second for the animation.
    colors : list of str
        Color for each class value, e.g., ["green", "red", "orange", "gray"] for class 0-3.
    class_labels : list of str
        Labels for the colorbar ticks, matching class values.
    """
    # ------------------------------------------------------------
    # Collect TIFF files
    # ------------------------------------------------------------
    tif_files = sorted(tiff_path.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError("No .tif files found in the directory.")

    dates = [f.stem for f in tif_files]  # Use file name for label

    # ------------------------------------------------------------
    # Prepare color map for classification
    # ------------------------------------------------------------
    if colors is None:
        colors = ["green", "red", "orange", "gray"]  # 0 = Clear, 1 = Cloudy, etc.

    n_classes = len(colors)
    cmap = ListedColormap(colors)
    vmin = 0
    vmax = n_classes - 1

    # ------------------------------------------------------------
    # Plot first frame
    # ------------------------------------------------------------
    with rasterio.open(tif_files[0]) as src:
        first = src.read(1)

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(first, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, ax=ax, ticks=list(range(n_classes)), fraction=0.046, pad=0.04)
    cbar.set_label("Cloud State")

    if class_labels:
        cbar.ax.set_yticklabels(class_labels)

    txt = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        color="white", fontsize=11, ha="left", va="top",
        bbox=dict(facecolor="black", alpha=0.3, pad=2, lw=0),
    )
    ax.set_axis_off()

    # ------------------------------------------------------------
    # Update function per frame
    # ------------------------------------------------------------
    def update(i: int):
        with rasterio.open(tif_files[i]) as src:
            arr = src.read(1)
        img.set_data(arr)
        txt.set_text(f"Date: {dates[i]}")
        return img, txt

    anim = FuncAnimation(
        fig, update, frames=len(tif_files),
        interval=1000 / fps, blit=True,
    )

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------
    output_dir = output_path / "animation-output"
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"{output_name}.gif"

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.show(fig)

    print(f"GIF saved to: {gif_path}")


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def plot_cloud_category(
    mesh_grid_path,
    feature_col,
    title,
    ax: Optional[plt.Axes] = None,
    category_colors=None,
    category_labels=None,
    figsize: Tuple[int, int] = (8, 6),
    show_edges=False 
):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    mesh_grid = gpd.read_file(mesh_grid_path)

    categories = sorted(mesh_grid[feature_col].dropna().unique())

    # Default for cloud_category
    default_colors = {
        0: "skyblue",      # Clear
        1: "gray",         # Cloudy
        2: "orange",       # Mixed
        3: "lightgreen"    # Not set
    }
    default_labels = {
        0: "0 - Clear",
        1: "1 - Cloudy",
        2: "2 - Mixed",
        3: "3 - Not set (assumed clear)"
    }

    category_colors = category_colors or {k: default_colors.get(k, "lightgray") for k in categories}
    category_labels = category_labels or {k: default_labels.get(k, str(k)) for k in categories}

    cmap = mcolors.ListedColormap([category_colors[k] for k in categories])
    bounds = [k - 0.5 for k in categories] + [categories[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    mesh_grid.plot(
        column=feature_col,
        cmap=cmap,
        norm=norm,
        linewidth=0 if not show_edges else 0.2,
        edgecolor="white" if show_edges else None,
        ax=ax
    )

    legend_elements = [
        Patch(facecolor=category_colors[k], edgecolor='black', label=category_labels[k])
        for k in categories
    ]
    ax.legend(handles=legend_elements, title=feature_col, loc="lower left")
    ax.set_title(title, fontsize=12)
    ax.axis("off")



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Optional, Dict, Tuple

def plot_cloud_category_raster(
    arr: np.ndarray,
    title: str = "Cloud Category Map",
    category_colors: Optional[Dict[int, str]] = None,
    category_labels: Optional[Dict[int, str]] = None,
    ax: Optional[plt.Axes] = None,
    nodata_value: Optional[int] = None,
    legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    imshow_kwargs: Optional[dict] = None,
) -> Tuple[plt.Axes, Optional[plt.Axes]]:
    """
    Display a discrete raster map (e.g., cloud categories) with fixed colors per class.
    Automatically masks nodata and sets white background for cleaner visualization.

    Parameters
    ----------
    arr : np.ndarray
        2D array of integer class labels.
    title : str
        Plot title.
    category_colors : dict[int, str], optional
        Mapping from class ID to color.
    category_labels : dict[int, str], optional
        Mapping from class ID to legend label.
    ax : plt.Axes, optional
        Matplotlib axis to draw on.
    nodata_value : int or float, optional
        Value to treat as nodata and mask out.
    legend : bool
        Whether to show legend.
    legend_kwargs : dict, optional
        Extra kwargs for ax.legend().
    imshow_kwargs : dict, optional
        Extra kwargs for imshow.

    Returns
    -------
    ax : plt.Axes
        The axes used for plotting.
    legend_ax : plt.Axes or None
        The legend, if created.
    """

    # === Handle category defaults ===
    if category_colors is None:
        category_colors = {
            0: "skyblue",     # Clear
            1: "gray",        # Cloudy
            2: "orange",      # Mixed
            3: "lightgreen"   # Not set
        }
    if category_labels is None:
        category_labels = {
            0: "0 - Clear",
            1: "1 - Cloudy",
            2: "2 - Mixed",
            3: "3 - Not set"
        }

    sorted_keys = sorted(category_colors.keys())
    cmap = mcolors.ListedColormap([category_colors[k] for k in sorted_keys])
    bounds = [k - 0.5 for k in sorted_keys] + [sorted_keys[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # === Mask nodata ===
    if nodata_value is not None:
        arr = np.ma.masked_where(arr == nodata_value, arr)
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.ma.masked_invalid(arr)

    # === Create figure and axis ===
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    imshow_kwargs = imshow_kwargs or {}
    cmap.set_bad(color="white")  # Set background to white
    img = ax.imshow(arr, cmap=cmap, norm=norm, **imshow_kwargs)

    ax.set_title(title)
    ax.axis("off")

    legend_ax = None
    if legend:
        legend_kwargs = legend_kwargs or {}
        handles = [Patch(color=category_colors[k], label=category_labels[k]) for k in sorted_keys]
        ax.legend(handles=handles, loc="lower right", **legend_kwargs)

    return ax, legend_ax

import matplotlib.pyplot as plt
import rasterio
import numpy as np
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
from pathlib import Path

def plot_filled_vs_original_raster(
    original_tif: Path,
    filled_tif: Path,
    shapefile: Path,
    output_image: Path = None
) -> None:
    """
    Plot original vs filled raster clipped by a shapefile geometry and optionally save the image.

    Parameters
    ----------
    original_tif : Path
        Path to the original raster.
    filled_tif : Path
        Path to the filled raster.
    shapefile : Path
        Path to the shapefile used to crop the filled raster.
    output_image : Path or None
        If provided, save the comparison image to this path.
    """
    shapes = gpd.read_file(shapefile)
    geometry = [mapping(shapes.geometry.iloc[0])]

    with rasterio.open(original_tif) as src_before:
        before = src_before.read(1).astype(float)
        before[before == src_before.nodata] = np.nan

    with rasterio.open(filled_tif) as src_after:
        out_image, _ = mask(src_after, geometry, crop=True, nodata=np.nan)
        after = out_image[0].astype(float)
        after[after == src_after.nodata] = np.nan

    combined = np.concatenate([before[~np.isnan(before)], after[~np.isnan(after)]])
    vmin = np.percentile(combined, 2)
    vmax = np.percentile(combined, 98)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = "jet"

    im1 = axes[0].imshow(before, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label("Temperature")

    im2 = axes[1].imshow(after, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label("Temperature")

    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=300)
        print(f"Saved image to: {output_image.name}")

    plt.show()

def generate_LST_mesh_animation(data_folder, output_gif, layer_name="LST_day", column_name="LST_day_mean", cmap="YlOrRd", dpi=100, fps=6, city_label=""):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import imageio
    import numpy as np
    from pathlib import Path

    gpkg_files = sorted(data_folder.glob("*.gpkg"))
    all_values = []
    valid_files = []
    skipped = []

    for file in gpkg_files:
        try:
            gdf = gpd.read_file(file, layer=layer_name)
            values = gdf[column_name].replace([None], np.nan).dropna().values
            if len(values) > 0:
                all_values.extend(values)
                valid_files.append(file)
        except:
            skipped.append(file.name)

    if skipped:
        print("Skipped files:", skipped)

    vmin = np.percentile(all_values, 2)
    vmax = np.percentile(all_values, 98)

    temp_img_dir = output_gif.parent / "temp_frames"
    temp_img_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for i, file in enumerate(valid_files):
        date_str = file.stem.split("-")[-1]
        gdf = gpd.read_file(file, layer=layer_name)
        gdf = gdf.replace({column_name: {None: np.nan}}).dropna(subset=[column_name])

        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(column=column_name, cmap=cmap, ax=ax, legend=False, vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.01).set_label("Surface Temperature (°C)", fontsize=10)
        ax.set_title(f"{city_label} LST on {date_str}", fontsize=12)
        ax.axis("off")

        img_path = temp_img_dir / f"frame_{i:03d}.png"
        plt.savefig(img_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        frame_paths.append(img_path)

    with imageio.get_writer(output_gif, mode='I', duration=1/fps) as writer:
        for img_path in frame_paths:
            writer.append_data(imageio.v3.imread(img_path))


from matplotlib.colors import ListedColormap, BoundaryNorm

def get_modis_landcover_colormap():
    """
    Return MODIS (IGBP) land cover classification values, names, colors, and normalization.

    Returns
    -------
    tuple: (class_values, class_names, cmap, norm)
    """
    import matplotlib.colors as mcolors

    class_info = {
        0:  ("Water", "#1f78b4"),
        1:  ("Evergreen Needleleaf Forest", "#005100"),
        2:  ("Evergreen Broadleaf Forest", "#008000"),
        3:  ("Deciduous Needleleaf Forest", "#339900"),
        4:  ("Deciduous Broadleaf Forest", "#66cc00"),
        5:  ("Mixed Forest", "#8db400"),
        6:  ("Closed Shrublands", "#ccae62"),
        7:  ("Open Shrublands", "#dcd159"),
        8:  ("Woody Savannas", "#c8b75a"),
        9:  ("Savannas", "#e2c68c"),
        10: ("Grasslands", "#f7e084"),
        11: ("Permanent Wetlands", "#4fa3cc"),
        12: ("Croplands", "#ffff64"),
        13: ("Urban and Built-Up", "#ff0000"),
        14: ("Cropland/Natural Vegetation Mosaic", "#bfbf00"),
        15: ("Snow and Ice", "#ffffff"),
        16: ("Barren or Sparsely Vegetated", "#dcdcdc"),
        254:("Unclassified", "#ffffff"),  # white for unclassified
        255:("Fill Value", "#ffffff")     # white for no data
    }

    class_values = list(class_info.keys())
    class_names = [class_info[val][0] for val in class_values]
    colors = [class_info[val][1] for val in class_values]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(class_values + [256], cmap.N)

    return class_values, class_names, cmap, norm




import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio

from matplotlib.colors import ListedColormap, BoundaryNorm

def get_esa_landcover_colormap():
    """
    Returns ESA WorldCover 2021 class values, names, colormap, and normalization.

    Returns
    -------
    class_values : list of int
        ESA land cover class codes.
    class_names : list of str
        Descriptions for each class.
    cmap : ListedColormap
        Color map for ESA land cover categories.
    norm : BoundaryNorm
        Value-to-color mapping.
    """
    class_values = [
        10, 20, 30, 40, 50, 60, 70, 80,
        90, 95, 100, 111, 112, 200
    ]

    class_names = [
        "Tree cover", "Shrubland", "Grassland", "Cropland",
        "Built-up", "Bare/Sparse Vegetation", "Snow and Ice", "Water Bodies",
        "Wetlands", "Mangroves", "Moss and Lichen", "Permanent Snow", "Glaciers", "No Data"
    ]

    class_colors = [
        "#006400",  # 10 Tree cover
        "#FFBB22",  # 20 Shrubland
        "#FFFF4C",  # 30 Grassland
        "#F096FF",  # 40 Cropland
        "#FA0000",  # 50 Built-up
        "#B4B4B4",  # 60 Bare/sparse vegetation
        "#F0F0F0",  # 70 Snow and ice
        "#0064C8",  # 80 Water bodies
        "#0096A0",  # 90 Wetlands
        "#00CF75",  # 95 Mangroves
        "#FAE6A0",  # 100 Moss and Lichen
        "#DCDCDC",  # 111 Permanent snow
        "#B0E0E6",  # 112 Glaciers
        "#FFFFFF"   # 200 No data → white
    ]

    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(class_values + [201], cmap.N)

    return class_values, class_names, cmap, norm

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.patches as mpatches

def plot_landcover_legend_map(
    tiff_path,
    class_values,
    class_names,
    cmap,
    norm,
    title="Land Cover Classification",
    figsize=(10, 10),
    legend_cols=3
):
    """
    Plot a categorical land cover map with a corresponding legend.

    Parameters
    ----------
    tiff_path : Path or str
        Path to the categorical land cover raster (.tif).
    class_values : list
        List of land cover class values (int).
    class_names : list
        Corresponding list of class names (str).
    cmap : ListedColormap
        Color map for displaying land cover classes.
    norm : BoundaryNorm
        Normalization to map values to colors.
    title : str
        Title of the plot.
    figsize : tuple
        Size of the matplotlib figure.
    legend_cols : int
        Number of columns in the legend.
    """
    with rasterio.open(tiff_path) as src:
        image = src.read(1)
        image = np.ma.masked_equal(image, 255)  # mask nodata

        fig, ax = plt.subplots(figsize=figsize)
        show(image, ax=ax, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.axis("off")

        # Build legend manually
        handles = [
            mpatches.Patch(color=cmap(norm(val)), label=name)
            for val, name in zip(class_values, class_names)
            if val in np.unique(image)
        ]

        # Add legend outside plot
        fig.legend(
            handles=handles,
            loc='lower center',
            ncol=legend_cols,
            bbox_to_anchor=(0.5, -0.05),
            fontsize=9
        )

        plt.tight_layout()
        plt.show()
