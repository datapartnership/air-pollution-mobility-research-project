from pathlib import Path
from typing import Union, List

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm

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
    # Helper to update each frame
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


def mesh_2_gif(
        gpkg_path: Path, 
        output_path: Path,
        output_name: str = "timeseries", 
        column: Union[str, None] = None,
        fps: int = 12,
        colors: Union[List[str], None] = None,
        percent_clip: float = 0.5
):
    
    """
    Read all .gpkg files from a directory, visualize them by a numeric column, and generate a time-series GIF animation.

    Parameters
    ----------
    gpkg_path : Path
        Directory containing .gpkg files.
    output_path: Path
        Directory that will save the output files.
    out_gif_name: str
        Output GIF filename (without extension).
    column : str or None
        The column to be visualized. If None, the first numeric column will be automatically selected.
    fps : int
        Frames per second in the GIF.
    colors : list of str or None
        List of color names for custom colormap. If None, defaults to ["blue", "cyan", "yellow", "red"].
    percent_clip : float
        Percentage of data to clip at both tails (e.g., 0.5 clips bottom 0.5% and top 0.5%).
    """
    
    # ------------------------------------------------------------
    # Collect files
    # ------------------------------------------------------------
    gpkg_files = sorted(gpkg_path.glob("*.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError("No .gpkg files found in the provided directory.")

    # Scan all files to determine global min/max for color normalization
    numeric_vals = []
    candidates = None
    for f in tqdm(gpkg_files, desc="Scanning percentiles"):
        gdf = gpd.read_file(f)

        # automatic or specified color selection
        if column is None:
            if candidates is None:
                candidates = gdf.select_dtypes(include=[np.number]).columns.tolist()
                if not candidates:
                    raise ValueError(f"No numeric columns found in {f.name}. Please specify 'column'.")
                column_auto = candidates[0]
            column_use = column_auto
        else:
            column_use = column
            if column_use not in gdf.columns:
                raise ValueError(f"Column '{column_use}' not found in {f.name}.")
            
        # collect data and clip the tips
        numeric_vals.extend(gdf[column_use].dropna().to_list())

    vmin = np.percentile(numeric_vals, percent_clip)
    vmax = np.percentile(numeric_vals, 100 - percent_clip)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # define the Colormap
    if colors is None:
        colors = ["blue", "cyan", "yellow", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_map", colors)

    # plot the first frame and update function
    first_gdf = gpd.read_file(gpkg_files[0])
    column_use = column_auto if column is None else column

    title_fmt = "Date: {date}"

    # frame update function
    def update(i):
        gdf = gpd.read_file(gpkg_files[i])
        ax.clear()
        gdf.plot(column=column_use, ax=ax, cmap=cmap, norm=norm, 
                 linewidth=0, legend=False)
        ax.set_axis_off()
        txt.set_text(title_fmt.format(date=gpkg_files[i].stem))
        return ax, txt

    fig, ax = plt.subplots(figsize=(6, 6))
    first_gdf.plot( column=column_use, ax=ax, cmap=cmap, norm=norm, 
                   linewidth=0, legend=False)
    ax.set_axis_off()
    txt = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        color="white", fontsize=11, ha="left", va="top",
        bbox=dict(facecolor="black", alpha=0.3, pad=2, lw=0)
    )

    # Generate animation
    anim = FuncAnimation(fig, update, frames=len(gpkg_files),
        interval=1000 / fps, blit=False)

    # Save animation
    output_dir = output_path / "animation-output"
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"{output_name}.gif"

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.show()

    print(f"Animation saved to: {gif_path}")