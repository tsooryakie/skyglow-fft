import os
import shutil
import raster_preprocessing
import skyglow_calculation
import skyglow_utils
from tqdm import tqdm

MODEL_SETTINGS = skyglow_utils.parse_model_settings("../settings/settings.toml")
KEEP_INTERMEDIATES = True


def main() -> None:
    skyglow_utils.set_up_directories(MODEL_SETTINGS)
    viirs_rasters_to_preprocess = sorted(
        [
            f"{MODEL_SETTINGS['rasters']['raster_read_path']}{raster}"
            for raster in os.listdir(MODEL_SETTINGS["rasters"]["raster_read_path"])
        ]
    )

    for viirs_raster in viirs_rasters_to_preprocess:
        raster_preprocessing.reproject_to_utm(
            viirs_path=viirs_raster,
            epsg_code=MODEL_SETTINGS["raster_preprocessing"]["epsg_code"],
            write_directory=MODEL_SETTINGS["product_directories"]["preprocessed"],
        )

    preprocessed_viirs_rasters = sorted(
        [
            f"{MODEL_SETTINGS['product_directories']['preprocessed']}{raster}"
            for raster in os.listdir(
                MODEL_SETTINGS["product_directories"]["preprocessed"]
            )
        ]
    )

    for viirs_raster in preprocessed_viirs_rasters:
        raster_preprocessing.downsample(
            epsg_code=MODEL_SETTINGS["raster_preprocessing"]["epsg_code"],
            viirs_path=viirs_raster,
            downsampled_raster_write_path=f"{viirs_raster.split('.tif')[0]}_5km.tif",
            factor=MODEL_SETTINGS["raster_preprocessing"]["downscaling_factor"],
        )

    for raster in preprocessed_viirs_rasters:
        if os.path.exists(raster):
            os.remove(raster)
        else:
            continue

    downsampled_rasters = sorted(
        [
            f"{MODEL_SETTINGS['product_directories']['preprocessed']}{raster}"
            for raster in os.listdir(
                MODEL_SETTINGS["product_directories"]["preprocessed"]
            )
        ]
    )

    for raster in downsampled_rasters:
        downsampled_viirs_array = skyglow_utils.load_raster_as_array(
            path_to_raster=raster
        )
        theta_array, distance_kernel = skyglow_calculation.compute_kernels(
            downsampled_viirs_path=raster
        )
        decayed_distance_kernel = skyglow_calculation.compute_decayed_distance_kernel(
            distance_kernel=distance_kernel, model_settings=MODEL_SETTINGS
        )

        print(f"Calculating Skyglow Segments for: {raster.split('/')[-1]}")
        for angle in tqdm(range(-180, 180, 10)):
            segment = skyglow_calculation.calculate_segments(
                theta_array=theta_array,
                angle=angle,
                decayed_distance_kernel=decayed_distance_kernel,
            )
            combined_frequency_shift = skyglow_calculation.compute_magnitude_spectrum(
                downsampled_viirs=downsampled_viirs_array,
                distance_kernel=segment,
                visualise=MODEL_SETTINGS["parameters"]["visualise"],
            )
            inverse_frequency_shift = skyglow_utils.compute_inverse_frequency_shifts(
                combined_frequency_shift=combined_frequency_shift
            )
            skyglow_calculation.write_raster(
                downsampled_viirs_path=raster,
                raster_write_path=f"{MODEL_SETTINGS['product_directories']['intermediates']}"
                f"skyglow_segment_{angle}.tif",
                inverse_frequency_shift=inverse_frequency_shift,
            )

        summed_segments = skyglow_calculation.sum_kernels(
            visualise=MODEL_SETTINGS["parameters"]["visualise"],
            model_settings=MODEL_SETTINGS,
        )
        skyglow_raster_write_path = (
            f"{MODEL_SETTINGS['product_directories']['product_write_path']}"
            f"{raster.split('/')[-1].split('.tif')[0]}_skyglow_result.tif"
        )
        skyglow_calculation.write_skyglow_raster(
            downsampled_viirs_path=raster,
            path_to_write=skyglow_raster_write_path,
            summed_segments=summed_segments,
        )

    if not KEEP_INTERMEDIATES:
        skyglow_utils.purge_intermediates(
            preprocessed_rasters_path=MODEL_SETTINGS["product_directories"][
                "preprocessed"
            ],
            intermediates_path=MODEL_SETTINGS["product_directories"]["intermediates"],
        )

    return


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        shutil.rmtree(MODEL_SETTINGS["product_directories"]["preprocessed"])
        shutil.rmtree(MODEL_SETTINGS["product_directories"]["intermediates"])
        shutil.rmtree(MODEL_SETTINGS["product_directories"]["product_write_path"])
        raise exception
