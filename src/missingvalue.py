# In this script, missing data in each raster is filled using the neighbour data, 
# and the filled raster is saved in a new seperate folder: Cityname_no2_filled

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from scipy.ndimage import generic_filter


def read_tiff(
        filename: str
) -> Tuple[DatasetReader, np.ndarray, dict, Optional[float]]:
    """
    Read a GeoTIFF file and return the raster band, metadata, and nodata value.

    Parameters
    ----------
    filename : str
        Path to the GeoTIFF file.

    Returns
    -------
    src : DatasetReader
        Rasterio dataset object.
    band : np.ndarray
        First band data from the raster.
    profile : dict
        Metadata profile of the raster.
    nodata_value : float or None
        Nodata value indicating missing data, if present.
    """

    with rasterio.open(filename) as src:
        band = src.read(1)
        profile = src.profile
        nodata_value = src.nodata
    return src, band, profile, nodata_value


def fill_nan_with_mean(
        arr: np.ndarray
) -> float:
    """
    Replace the center pixel with the mean of the neighbour cells if it is NaN.

    Parameters
    ----------
    arr : np.ndarray
        The neighborhood array around a pixel.

    Returns
    -------
    float
        Replaced value for the center pixel.
    """
    center = arr[len(arr) // 2]
    return np.nanmean(arr) if np.isnan(center) else center


def iterative_fill(
    data: np.ndarray,
    max_iter: int = 10,
    window_size: int = 9
) -> np.ndarray:
    """
    Iteratively fill NaN values using a moving average window.

    Parameters
    ----------
    data : np.ndarray
        2D array with NaN values.
    max_iter : int
        Maximum number of iterations to perform.
    window_size : int
        Size of the moving window, should be an odd number.

    Returns
    -------
    np.ndarray
        Filled 2D array.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number to avoid raster shifting.")
    
    filled = data.copy()
    for i in range(max_iter):
        prev_nan_count = np.isnan(filled).sum()
        filled = generic_filter(filled, function=fill_nan_with_mean, size=window_size, mode='nearest')
        new_nan_count = np.isnan(filled).sum()
        if new_nan_count == 0 or new_nan_count == prev_nan_count:
            break
    return filled


def fill_missing_data(
    country: str,
    data_tiff_path: Path,
    output_path: Path
) -> None:
    """
    Fill missing (nodata) values in all TIFF files in a given directory and
    save the filled files to a subdirectory under the given output root.

    Parameters
    ----------
    country : str
        Name of the country (first letter uppercase), e.g., 'Iraq'.
    data_tiff_path : Path
        Path to the folder containing input TIFF files.
    output_path : Path
        Root path where the output folder '{country}-no2-filled' will be created.

    Returns
    -------
    None
        Processed TIFF files are saved to the output directory under output_path.
    """
    # Collect all .tif files
    tiff_files = sorted([f for f in data_tiff_path.glob("*.tif")])
    n_task = len(tiff_files)

    # Define output directory and create it if not exist
    output_dir = output_path / f"{country}-no2-filled"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tiff_path in enumerate(tiff_files):
        date = tiff_path.stem.split('_')[2]
        print(f"Processing {index + 1}/{n_task}: {date}")

        file_size_mb = tiff_path.stat().st_size / (1024 * 1024)
        if file_size_mb < 1:
            print(f"Skipping {date}: file size {file_size_mb:.2f}MB < 1MB.")
            continue

        src, band, profile, nodata_value = read_tiff(tiff_path)
        if nodata_value is not None:
            band = np.where(band == nodata_value, np.nan, band)

        band_filled = iterative_fill(band, max_iter=10, window_size=9)

        output_file = output_dir / f"{country}_NO2_{date}_filled.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            filled_band = np.where(np.isnan(band_filled), nodata_value, band_filled)
            dst.write(filled_band.astype(profile['dtype']), 1)
