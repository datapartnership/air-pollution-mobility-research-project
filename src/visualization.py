# This script defines the functions that are used to generate related visualization,
# including:
#   (1) static plots: plot_percent_clipped_raster()
#   (2) dynamic plots (animation): tiff_2_gif(), mesh_2_gif()


import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from typing import List, Union, Tuple, Optional
from pathlib import Path


def plot_raster(
    arr: np.ndarray,
    percent_clip: float = 0.5,
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
    **plot_kwargs :
        Additional keyword arguments passed to `GeoDataFrame.plot()`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    mesh.plot(column=feature, edgecolor="grey", legend=True, ax=ax, **plot_kwargs)
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