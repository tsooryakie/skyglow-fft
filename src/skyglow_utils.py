import shutil
import rasterio as rio
import numpy as np
from rasterio.warp import Resampling, transform, reproject
from typing import Dict


def load_raster_as_array(path_to_raster: str) -> np.ndarray:
    """
    This function uses Rasterio to load the VIIRS raster as an Numpy array.
    :param path_to_raster: Path to the VIIRS raster to be used for skyglow calculation
    :return raster: VIIRS raster as a Numpy array
    """

    with rio.open(path_to_raster, mode="r") as source:
        raster = source.read(1)

        source.close()

    return raster


def write_raster_reprojection(
    path_to_write: str, epsg_code: str, metadata: Dict, source: rio.DatasetReader
) -> None:
    """
    This function writes the reprojected raster to disk.
    :param path_to_write: Path to which the raster is to be written to
    :param source: Source projection raster (original VIIRS, before reprojection)
    :param metadata: Source projection raster metadata.
    :param epsg_code: Coordinate system code
    :return:
    """

    with rio.open(path_to_write, mode="w", **metadata) as dst:
        reproject(
            source=rio.band(source, 1),
            destination=rio.band(dst, 1),
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=epsg_code,
            resampling=Resampling.nearest,
        )

    return


def apply_decay_function(distance_kernel: np.ndarray) -> np.ndarray:
    """
    This function applies the skyglow decay function to the distance kernel.
    :param distance_kernel: Array of distances from the center of the array
    :return:
    """

    return 27.97202 * np.exp(-0.00823 * (distance_kernel / 1000))


def compute_frequency_shifts(array: np.ndarray) -> np.ndarray:
    """
    This function computes the frequency shift of a given array (e.g. VIIRS raster or distance kernel),
    and hence takes the array from spatial domain into a frequency domain.
    :param array: Array in spatial domain
    :return:
    """
    frequency = np.fft.fft2(array)
    frequency_shift = np.fft.fftshift(frequency)

    return frequency_shift


def compute_inverse_frequency_shifts(
    combined_frequency_shift: np.ndarray,
) -> np.ndarray:
    """
    This function computes the inverse frequency shift of a given array and transforms the
    combined VIIRS and segment kernel, from frequency domain back into spatial domain.
    :param combined_frequency_shift: Combined Frequency Shift of distance kernel and VIIRS image
    :return inverse_frequency_shift: Inverse of the combined Frequency Shift
    """

    inverse_frequency = np.fft.ifft2(combined_frequency_shift)
    inverse_frequency_shift = np.abs(np.fft.ifftshift(inverse_frequency))

    return inverse_frequency_shift


def purge_intermediates() -> None:
    """
    This function purges the intermediate segment rasters created during skyglow calculation.
    :return:
    """

    shutil.rmtree("../intermediates")

    return
