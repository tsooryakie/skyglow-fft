import numpy as np
import matplotlib.pyplot as plt


def log_transformation(frequency_domain_array: np.ndarray) -> np.ndarray:
    """
    This function will undertake a log transformation of an array in frequency domain,
    into a visualisable array.
    :param frequency_domain_array: Non-visualisable array in frequency domain
    :return log_transformed_array: Visualisable array under log-transformation
    """
    return 20 * np.log(np.abs(frequency_domain_array))


def visualise_transforms(
    viirs_raster: np.ndarray,
    viirs_log_spectrum: np.ndarray,
    distance_kernel: np.ndarray,
    distance_log_spectrum: np.ndarray,
    combined_frequency_shift: np.ndarray,
) -> None:
    """
    This function creates visualisations of the arrays supplied in arguments in both spatial and frequency domains.
    In order to visualise the arrays in frequency domains, the arrays undergo a log-transformation of their
    absolute values.
    :param viirs_raster: VIIRS raster array
    :param viirs_log_spectrum: VIIRS raster array in Frequency domain (under log-transformation)
    :param distance_kernel: Distance kernel array
    :param distance_log_spectrum: Distance kernel array in Frequency domain (under log-transformation)
    :param combined_frequency_shift: Combined Frequency Shift array (under log-transformation)
    :return: Void
    """

    plt.subplot(121), plt.imshow(viirs_raster, cmap="gist_gray", vmin=0, vmax=1)
    plt.title("VIIRS image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(viirs_log_spectrum, cmap="gist_gray")
    plt.title("VIIRS Frequency Shift"), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(121), plt.imshow(distance_kernel, cmap="gist_gray", vmin=0, vmax=1)
    plt.title("Distance Kernel"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(distance_log_spectrum, cmap="gist_gray")
    plt.title("Distance Kernel Frequency Shift"), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.imshow(combined_frequency_shift, cmap="gist_gray")
    plt.title("Combined Frequency Shifts (Log Transformed)")
    plt.show()

    return
