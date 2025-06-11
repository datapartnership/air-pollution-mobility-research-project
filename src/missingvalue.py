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
    max_iter: int = 5,
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


from datetime import datetime, timedelta
import calendar

def day_number_to_date(year, day_number):
    is_leap = calendar.isleap(year)
    max_days = 366 if is_leap else 365

    if 1 <= day_number <= max_days:
        date = datetime(year, 1, 1) + timedelta(days=day_number - 1)
        return date.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"{day_number} exceed {year} maximum {max_days}")

    
def fill_ntl_missing_data(
    city: str,
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
    output_dir = output_path / f"{city}-NTL-filled"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tiff_path in enumerate(tiff_files):
        date_str = tiff_path.stem.split('.')[1]
        year = int(date_str[1:5])      # '2023' -> 2023
        day = int(date_str[5:8])
        date = day_number_to_date(year, day)
        print(f"Processing {index + 1}/{n_task}: {date}")

        # file_size_mb = tiff_path.stat().st_size / (1024 * 1024)
        # if file_size_mb < 1:
        #     print(f"Skipping {date}: file size {file_size_mb:.2f}MB < 1MB.")
        #     continue

        src, band, profile, nodata_value = read_tiff(tiff_path)
        if nodata_value is not None:
            band = np.where(band == nodata_value, np.nan, band)

        band_filled = iterative_fill(band, max_iter=10, window_size=9)

        output_file = output_dir / f"{city}_NTL_{date}_filled.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            filled_band = np.where(np.isnan(band_filled), nodata_value, band_filled)
            dst.write(filled_band.astype(profile['dtype']), 1)

from pathlib import Path
import numpy as np
import rasterio
from missingvalue import read_tiff, iterative_fill

def fill_surface_temperature_data(
    city: str,
    data_tiff_path: Path,
    output_path: Path
) -> None:
    """
    Fill NaN values in MODIS LST GeoTIFFs, using iterative window-based mean filtering.
    If the entire image is NaN, it is kept as NaN for downstream identification.

    Parameters
    ----------
    city : str
        City name (used for naming the output folder and files).
    data_tiff_path : Path
        Folder containing the input TIFF files.
    output_path : Path
        Root folder to save the output (a subfolder {city}-LST-filled will be created).

    Output
    -------
    GeoTIFFs with missing values filled or kept as NaN.
    File naming format: {city}_LST_YYYY-MM-DD_filled.tif
    """

    tiff_files = sorted(data_tiff_path.glob("*.tif"))
    output_dir = output_path / f"{city}-LST-filled"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tiff_path in enumerate(tiff_files):
        # Extract date from filename
        parts = tiff_path.stem.split("_")
        try:
            date = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
        except IndexError:
            print(f"[Skipped] Unable to parse date from filename: {tiff_path.name}")
            continue

        print(f"[{index + 1}/{len(tiff_files)}] Processing date: {date}")

        src, band, profile, nodata_value = read_tiff(str(tiff_path))

        # Replace nodata_value with NaN for processing
        if nodata_value is not None:
            band = np.where(band == nodata_value, np.nan, band)
        else:
            nodata_value = -9999  # fallback
        profile.update(nodata=nodata_value)

        # Fill missing data
        band_filled = iterative_fill(band, max_iter=10, window_size=9)

        # Case 1: Entire image is NaN
        if np.isnan(band_filled).all():
            print(f"⚠️ Entire image is NaN: {tiff_path.name}")
            filled_band = band_filled
        else:
            # Case 2: Fill remaining NaNs with global mean
            global_mean = np.nanmean(band_filled)
            band_filled = np.where(np.isnan(band_filled), global_mean, band_filled)
            filled_band = np.where(np.isnan(band_filled), nodata_value, band_filled)

        # Save the output
        output_file = output_dir / f"{city}_LST_{date}_filled.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(filled_band.astype(profile['dtype']), 1)

from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from scipy.stats import mode

def fill_nan_with_mode(window: np.ndarray) -> float:
    """
    Replace the center pixel with the mode of its neighbors if it is NaN.
    """
    center = window[len(window) // 2]
    if not np.isnan(center):
        return center
    neighbors = window[~np.isnan(window)].astype(int)
    return mode(neighbors, keepdims=False).mode[0] if len(neighbors) > 0 else np.nan

def iterative_fill_categorical(data: np.ndarray, max_iter: int = 10, window_size: int = 9) -> np.ndarray:
    """
    Iteratively fill NaN in a categorical raster using neighborhood mode.

    Parameters
    ----------
    data : np.ndarray
        2D array with NaN values.
    max_iter : int
        Maximum number of iterations.
    window_size : int
        Size of the neighborhood window (must be odd).
    """
    filled = data.copy()
    for _ in range(max_iter):
        previous_nan = np.isnan(filled).sum()
        filled = generic_filter(filled, function=fill_nan_with_mode, size=window_size, mode='nearest')
        if np.isnan(filled).sum() == 0 or np.isnan(filled).sum() == previous_nan:
            break
    return filled

def fill_landcover_data(
    input_tiff_path: Path,
    output_tiff_path: Path,
    default_nodata: int = 255
) -> None:
    """
    Fill missing values in a land cover GeoTIFF using neighborhood mode.

    Parameters
    ----------
    input_tiff_path : Path
        Path to the input GeoTIFF file.
    output_tiff_path : Path
        Path to save the filled GeoTIFF.
    default_nodata : int
        Value used to represent nodata in the raster.
    """
    with rasterio.open(input_tiff_path) as src:
        band = src.read(1).astype(float)
        profile = src.profile
        nodata_val = src.nodata if src.nodata is not None else default_nodata

    band[band == nodata_val] = np.nan

    # Apply iterative mode-based filling
    filled_band = iterative_fill_categorical(band)

    # Replace any remaining NaN with global mode
    valid_values = filled_band[~np.isnan(filled_band)].astype(int)
    global_mode = mode(valid_values, keepdims=False).mode[0]
    filled_band = np.where(np.isnan(filled_band), global_mode, filled_band)

    # Write back to file
    profile.update(nodata=nodata_val)
    with rasterio.open(output_tiff_path, 'w', **profile) as dst:
        dst.write(filled_band.astype(profile["dtype"]), 1)
