# In this script, missing data in each raster is filled using the neighbour data, 
# and the filled raster is saved in a new seperate folder: Cityname_no2_filled

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from scipy.ndimage import generic_filter
import re



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
    output_path: Path,
    feature: str = "NO2"
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
    output_dir = output_path / f"{country}-{feature}-filled"
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

        output_file = output_dir / f"{country}_{feature}_{date}_filled.tif"
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


from scipy.ndimage import generic_filter

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

def fill_class_with_mode(data, nodata=np.nan):
    mask = np.isnan(data) if np.isnan(nodata) else (data == nodata)

    def mode_filter(values):
        vals = values[~np.isnan(values)]
        if len(vals) == 0:
            return np.nan
        return np.bincount(vals.astype(int)).argmax()

    filled = generic_filter(data, mode_filter, size=3, mode='constant', cval=np.nan)
    result = np.where(mask, filled, data)
    return result




def fill_cloud_missing_data(
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
        Root path where the output folder '{country}-cloud-filled' will be created.

    Returns
    -------
    None
        Processed TIFF files are saved to the output directory under output_path.
    """
    # Collect all .tif files
    tiff_files = sorted([f for f in data_tiff_path.glob("*.tif")])
    n_task = len(tiff_files)

    # Define output directory and create it if not exist
    output_dir = output_path / f"{city}-cloud-filled"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tiff_path in enumerate(tiff_files):
        date_str = tiff_path.stem.split('_')[2]
        year = int(date_str[0:4])      # '2023' -> 2023
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        date = datetime(year, month, day).date()
        print(f"Processing {index + 1}/{n_task}: {date}")

        src, band, profile, nodata_value = read_tiff(tiff_path)
        if nodata_value is not None:
            band = np.where(band == nodata_value, np.nan, band)

        band_filled = fill_class_with_mode(band, nodata=nodata_value)

        output_file = output_dir / f"{city}_Cloud_{date}_filled.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            filled_band = np.where(np.isnan(band_filled), nodata_value, band_filled)
            dst.write(filled_band.astype(profile['dtype']), 1)


def fill_temp_missing_data(
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
        Root path where the output folder '{country}-temp-filled' will be created.

    Returns
    -------
    None
        Processed TIFF files are saved to the output directory under output_path.
    """
    # Collect all .tif files
    tiff_files = sorted([f for f in data_tiff_path.glob("*.tif")])
    n_task = len(tiff_files)

    # Define output directory and create it if not exist
    output_dir = output_path / f"{city}-temp-filled"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tiff_path in enumerate(tiff_files):
        parts = tiff_path.stem.split('_')  # ['MODIS', 'LST', 'Day', '2023', '01', '01']
        date_str = f"{parts[3]}{parts[4]}{parts[5]}"
        year = int(date_str[0:4])      # '2023' -> 2023
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        date = datetime(year, month, day).date()
        print(f"Processing {index + 1}/{n_task}: {date}")

        src, band, profile, nodata_value = read_tiff(tiff_path)
        if nodata_value is not None:
            band = np.where(band == nodata_value, np.nan, band)

        band_filled = iterative_fill(band, max_iter=10, window_size=9)

        output_file = output_dir / f"{city}_temp_{date}_filled.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            filled_band = np.where(np.isnan(band_filled), nodata_value, band_filled)
            dst.write(filled_band.astype(profile['dtype']), 1)

from scipy.ndimage import generic_filter

def fill_class_with_mode(data, nodata=np.nan):
    mask = np.isnan(data) if np.isnan(nodata) else (data == nodata)

    def mode_filter(values):
        vals = values[~np.isnan(values)]
        if len(vals) == 0:
            return np.nan
        return np.bincount(vals.astype(int)).argmax()

    filled = generic_filter(data, mode_filter, size=3, mode='constant', cval=np.nan)
    result = np.where(mask, filled, data)
    return result


def fill_cloud_missing_data(
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
        Root path where the output folder '{country}-cloud-filled' will be created.

    Returns
    -------
    None
        Processed TIFF files are saved to the output directory under output_path.
    """
    # Collect all .tif files
    tiff_files = sorted([f for f in data_tiff_path.glob("*.tif")])
    n_task = len(tiff_files)

    # Define output directory and create it if not exist
    output_dir = output_path / f"{city}-cloud-filled"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, tiff_path in enumerate(tiff_files):
        date_str = tiff_path.stem.split('_')[2]
        year = int(date_str[0:4])      # '2023' -> 2023
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        date = datetime(year, month, day).date()
        print(f"Processing {index + 1}/{n_task}: {date}")

        src, band, profile, nodata_value = read_tiff(tiff_path)
        if nodata_value is not None:
            band = np.where(band == nodata_value, np.nan, band)

        band_filled = fill_class_with_mode(band, nodata=nodata_value)

        output_file = output_dir / f"{city}_Cloud_{date}_filled.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            filled_band = np.where(np.isnan(band_filled), nodata_value, band_filled)
            dst.write(filled_band.astype(profile['dtype']), 1)





import os
import re
from pathlib import Path
import rasterio
import numpy as np
from scipy.ndimage import generic_filter

def fill_surface_temperature_data(city, data_tiff_path, output_path):
    """
    Fill missing values in daily MODIS LST TIFFs using 3x3 mean filter.

    Parameters
    ----------
    city : str
        City identifier (e.g. 'baghdad', 'addis-ababa').
    data_tiff_path : Path
        Path to the folder containing original daily LST TIFF files.
    output_path : Path
        Folder to save filled TIFFs (with consistent naming).
    """
    tif_files = sorted([f for f in os.listdir(data_tiff_path) if f.endswith(".tif")])
    print(f"Detected city: {city.split('-')[0]}")  # Debug print

    for file in tif_files:
        full_path = data_tiff_path / file
        try:
            with rasterio.open(full_path) as src:
                arr = src.read(1)
                nodata = src.nodata if src.nodata is not None else -9999

                def mode_filter(values):
                    vals = values[values != nodata]
                    if len(vals) == 0:
                        return nodata
                    return np.nanmean(vals)

                filled = generic_filter(arr, mode_filter, size=3, mode='constant', cval=nodata)

                out_meta = src.meta.copy()
                out_meta.update({"nodata": nodata})

            # Extract date using regex (supports _ or -)
            match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", file)
            if match:
                date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                out_name = f"{city}_LST_{date_str}_filled.tif"
                out_path = output_path / out_name

                with rasterio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(filled, 1)

            else:
                print(f" Cannot extract date from {file}, skipping.")
                continue

        except Exception as e:
            print(f" Failed to process {file}: {e}")
            continue

    print(f"✅ Surface temperature filling complete for {len(tif_files)} files in: {data_tiff_path}")


import rasterio
import numpy as np
from scipy import stats
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import Affine

def fill_landcover_data(input_tiff_path, output_tiff_path, default_nodata=255, block_size=512):
    """
    Fill nodata areas in a categorical raster using majority filtering from neighboring valid pixels.

    Parameters
    ----------
    input_tiff_path : str or Path
        Path to the input raster file with categorical values.
    output_tiff_path : str or Path
        Path to save the filled raster.
    default_nodata : int
        The nodata value used in the input raster.
    block_size : int
        Size of the sliding window to apply mode filtering.
    """
    with rasterio.open(input_tiff_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, nodata=default_nodata)
        data = src.read(1)

    # Prepare output array
    filled = data.copy()

    # Identify where nodata exists
    mask_nodata = (data == default_nodata)

    # Iterate over blocks and apply mode filtering
    rows, cols = data.shape
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            win = (slice(i, min(i + block_size, rows)), slice(j, min(j + block_size, cols)))
            block = data[win]

            if np.any(block == default_nodata):
                valid_pixels = block[block != default_nodata]
                if valid_pixels.size > 0:
                    mode_value = stats.mode(valid_pixels, axis=None, keepdims=False).mode
                    block[(block == default_nodata)] = mode_value
                    filled[win] = block

    # Save filled raster
    with rasterio.open(output_tiff_path, "w", **profile) as dst:
        dst.write(filled, 1)
