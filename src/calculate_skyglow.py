import os
import sys
import numpy as np
import rasterio as rio
import skyglow_utils
import matplotlib.pyplot as plt
import visualisation_utils as vis_utils
from tqdm import tqdm
from typing import Tuple, Any


def compute_kernels(downsampled_viirs_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function uses the downsampled VIIRS raster in UTM projection to compute kernels.
    Each kernel is multiplied by pixel size in X and Y dimensions to match the VIIRS pixel sizes.
    An array of distance to the middle of the kernel is also generated for the distance kernel.
    :param downsampled_viirs_path: Path to VIIRS raster (downsampled in UTM projection)
    :return theta, distance_kernel: Array of angles from middle of the kernel; kernel of distances from the middle.
    """

    with rio.open(downsampled_viirs_path, mode="r") as src:

        viirs_transform = src.get_transform()
        image_shape = (src.height, src.width)
        x_pixel_size = viirs_transform[1]
        y_pixel_size = viirs_transform[-1]

        src.close()

    x_kernel = np.zeros(image_shape)

    """
    Creates 2-D arrays for X and Y dimensions of the kernel.
    Both arrays are then filled with sequence of numbers from 0 to N and 
    from 0 to -N from the middle row/column (0) of the array, outwards.
    Changing anything here will most likely break the script... Terribly.
    """

    try:
        for i in range(int(x_kernel.shape[1] / 2)):
            x_kernel[:, int(x_kernel.shape[1] / 2) + i] = i

        x_kernel[:, -1] = int(x_kernel.shape[1] / 2)

        for i in range(0, int(x_kernel.shape[1] / 2), 1):
            x_kernel[:, i] = -int(x_kernel.shape[1] / 2) + i

    except IndexError:
        pass

    x_kernel = x_kernel * x_pixel_size

    y_kernel = np.zeros(image_shape)

    try:
        for i in range(int(y_kernel.shape[0] / 2)):
            y_kernel[int(y_kernel.shape[0] / 2) + i, :] = i

        for i in range(0, int(y_kernel.shape[0] / 2), 1):
            y_kernel[i, :] = -int(y_kernel.shape[0] / 2) + i

    except IndexError:
        pass

    y_kernel = y_kernel * y_pixel_size

    distance_kernel = np.sqrt((x_kernel ** 2 + y_kernel ** 2))
    theta_array = np.rad2deg(np.arctan2(x_kernel, y_kernel))
    theta_array = np.nan_to_num(x=theta_array, nan=0)

    return theta_array, distance_kernel


def compute_decayed_distance_kernel(distance_kernel: np.ndarray) -> np.ndarray:
    """
    This function generates the skyglow-decayed distance kernel via applying the decay function,
    and masks values over distances where skyglow would not be visible (in this case, 412km from source).
    :param distance_kernel: Original, non-decayed distance kernel
    :return decayed_distance_kernel: Distance kernel with decay function applied, and skyglow visibility fixed.
    """

    decayed_distance_kernel = skyglow_utils.apply_decay_function(
        distance_kernel=distance_kernel
    )
    decayed_distance_kernel[distance_kernel == 0] = 0.5
    decayed_distance_kernel[distance_kernel > 412000] = 0
    decayed_distance_kernel[distance_kernel < -412000] = 0
    decayed_distance_kernel[distance_kernel < 0] = 0
    decayed_distance_kernel = np.nan_to_num(decayed_distance_kernel, 0)

    return decayed_distance_kernel


def calculate_segments(
    theta_array: np.ndarray, angle: int, decayed_distance_kernel: np.ndarray
) -> np.ndarray:
    """
    This function calculates the angles for segments in 10 degree range within the distance kernel.
    It uses the theta array to create segments within the distance kernel,
    and computes the Fast Fourier Transform/Inverse Fast Fourier Transform with those segments as kernels.
    :param theta_array: Array of angles calculated from the center
    :param angle: The current angle range
    :param decayed_distance_kernel: Array of distance values from the middle of the array
    :return segments: 10 degree segments of the decayed distance kernel
    """

    segment = np.where(
        (theta_array > angle) & (theta_array < angle + 10), decayed_distance_kernel, 0
    )

    return segment


def compute_magnitude_spectrum(
    downsampled_viirs: np.ndarray, distance_kernel: np.ndarray, visualise: bool
) -> np.ndarray:
    """
    This function transforms the input VIIRS image and segment distance kernel from spatial domain,
    into frequency domain using a Fast Fourier Transform.
    :param downsampled_viirs: VIIRS raster array
    :param distance_kernel: Distance kernel array
    :param visualise: If true, produces visualisations of the transforms.
    :return combined_frequency_shift: Frequency Shifts of both VIIRS and distance kernel combined
    """

    viirs_frequency_shift = skyglow_utils.compute_frequency_shifts(
        array=downsampled_viirs
    )
    distance_frequency_shift = skyglow_utils.compute_frequency_shifts(
        array=distance_kernel
    )
    combined_frequency_shift = viirs_frequency_shift * distance_frequency_shift

    if visualise:
        vis_utils.visualise_transforms(
            viirs_raster=downsampled_viirs,
            viirs_log_spectrum=vis_utils.log_transformation(
                frequency_domain_array=viirs_frequency_shift
            ),
            distance_kernel=distance_kernel,
            distance_log_spectrum=vis_utils.log_transformation(
                frequency_domain_array=distance_frequency_shift
            ),
            combined_frequency_shift=vis_utils.log_transformation(
                frequency_domain_array=combined_frequency_shift
            ),
        )

    return combined_frequency_shift


def write_raster(
    downsampled_viirs_path: str,
    raster_write_path: str,
    inverse_frequency_shift: np.ndarray,
) -> None:
    """
    This function writes the skyglow raster for each 10 degree segment, based on the inverse frequency shift
    array, from which the skyglow is computed.
    :param downsampled_viirs_path: Path to a downsampled VIIRS image from which the georeference is used
    :param raster_write_path: Path to file which the skyglow segment is to be written to
    :param inverse_frequency_shift: Inverse frequency shift array for a given segment
    :return:
    """

    with rio.open(downsampled_viirs_path, mode="r") as src:
        profile = src.profile

    with rio.open(
        raster_write_path,
        mode="w",
        driver="GTiff",
        height=inverse_frequency_shift.shape[0],
        width=inverse_frequency_shift.shape[1],
        count=1,
        dtype=rio.dtypes.float64,
        crs=profile["crs"],
        transform=profile["transform"],
    ) as dst:

        dst.write(inverse_frequency_shift, 1)

    return


def sum_kernels(visualise: bool, write_to_raster: bool, *args: Any) -> np.ndarray:
    """
    This function sums the individual skyglow segments into a single array, which can be
    visualised and written as a standalone raster.
    :param visualise: If true, visualises the summed output
    :param write_to_raster: If true, saves the summed output as a raster
    :return:
    """

    skyglow_segment_rasters = os.listdir("../intermediates")

    skyglow_segments = []
    for skyglow_segment in skyglow_segment_rasters:
        skyglow_segments.append(
            skyglow_utils.load_raster_as_array(f"../intermediates/{skyglow_segment}")
        )

    summed_segments = sum(skyglow_segments)

    if visualise:
        plt.imshow(summed_segments)
        plt.show()

    return summed_segments


def write_skyglow_raster(
    downsampled_viirs_path: str, path_to_write: str, summed_segments: np.ndarray
) -> None:
    """
    This function writes the summed segments as a complete skyglow raster.
    :param downsampled_viirs_path: Downsampled VIIRS raster to copy georeference from
    :param path_to_write: Path to write the raster to
    :param summed_segments: Summed skyglow array
    :return:
    """

    with rio.open(downsampled_viirs_path, mode="r") as source:
        profile = source.profile

    with rio.open(
        path_to_write,
        mode="w",
        driver="GTiff",
        height=summed_segments.shape[0],
        width=summed_segments.shape[1],
        count=1,
        dtype=rio.dtypes.float64,
        crs=profile["crs"],
        transform=profile["transform"],
    ) as dst:
        dst.write(summed_segments, 1)

    return


def main() -> None:

    downsampled_viirs_path = sys.argv[1]
    downsampled_viirs_array = skyglow_utils.load_raster_as_array(
        path_to_raster=downsampled_viirs_path
    )
    theta_array, distance_kernel = compute_kernels(downsampled_viirs_path=sys.argv[1])
    decayed_distance_kernel = compute_decayed_distance_kernel(
        distance_kernel=distance_kernel
    )

    for angle in tqdm(range(-180, 180, 10)):
        segment = calculate_segments(
            theta_array=theta_array,
            angle=angle,
            decayed_distance_kernel=decayed_distance_kernel,
        )
        combined_frequency_shift = compute_magnitude_spectrum(
            downsampled_viirs=downsampled_viirs_array,
            distance_kernel=segment,
            visualise=False,
        )
        inverse_frequency_shift = skyglow_utils.compute_inverse_frequency_shifts(
            combined_frequency_shift=combined_frequency_shift
        )
        write_raster(
            downsampled_viirs_path=downsampled_viirs_path,
            raster_write_path=f"../intermediates/skyglow_segment{angle}.tif",
            inverse_frequency_shift=inverse_frequency_shift,
        )

    summed_segments = sum_kernels(visualise=True, write_to_raster=True)
    write_skyglow_raster(
        downsampled_viirs_path=downsampled_viirs_path,
        path_to_write="skyglow_raster.tif",
        summed_segments=summed_segments,
    )
    skyglow_utils.purge_intermediates()

    return


if __name__ == "__main__":
    main()
