import rasterio as rio
import skyglow_utils as utils
from rasterio import Affine
from rasterio.warp import calculate_default_transform


def reproject_to_utm(viirs_path: str, epsg_code: str, write_directory: str) -> None:
    """
    This function reprojects a VIIRS raster from native-WGS84 to a UTM CRS e.g UTM 32N
    :param viirs_path: path to VIIRS image
    :param epsg_code: EPSG coordinate system code
    :param write_directory: Directory to save reprojected raster to
    :return: Void
    """

    with rio.open(viirs_path, "r") as source:
        transform, width, height, = calculate_default_transform(
            source.crs, epsg_code, source.width, source.height, *source.bounds
        )

        raster_metadata = source.meta.copy()
        raster_metadata.update(
            {"crs": epsg_code, "transform": transform, "width": width, "height": height}
        )

        utils.write_raster_reprojection(
            path_to_write=f"{write_directory}{viirs_path.split('/')[-1].split('.tif')[0]}_utm.tif",
            epsg_code=epsg_code,
            metadata=raster_metadata,
            source=source,
        )

        source.close()

    return


def downsample(
    epsg_code: str, viirs_path: str, downsampled_raster_write_path: str, factor: int
) -> None:
    """
    This function downsamples the resolution of the input raster by a given factor.
    E.g. dimensions/7 downsamples the raster to ~5km per pixel resolution
    :param epsg_code: EPSG Coordinate System Code
    :param viirs_path: Path to the reprojected VIIRS image
    :param downsampled_raster_write_path: Path to write the downsampled raster to
    :param factor: Factor by which to downsample the raster
    :return: Void
    """

    with rio.open(viirs_path, mode="r") as source:
        transform, width, height, = calculate_default_transform(
            source.crs, epsg_code, source.width, source.height, *source.bounds
        )

        transform = Affine(
            transform.a * factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e * factor,
            transform.f,
        )

    with rio.open(viirs_path, mode="r") as source:
        data = source.read(
            1,
            out_shape=(int(source.height / factor), int(source.width / factor)),
            resampling=rio.enums.Resampling.average,
        )

        with rio.open(
            downsampled_raster_write_path,
            mode="w",
            driver="GTiff",
            height=int(source.height / factor),
            width=int(source.width / factor),
            count=1,
            dtype=rio.dtypes.float32,
            crs=epsg_code,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        source.close()

    return
