import os
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from glob import glob


def reproject_to_utm(viirs: str, epsg_code: str):
    """
    This function reprojects a VIIRS raster from native-WGS84 to a UTM CRS e.g UTM 32N
    :param viirs: path to VIIRS image
    :param epsg_code: EPSG coordinate system code
    :return:
    """

    dst_crs = epsg_code #destination Coordinate Reference System - refer to EPSG for CRS codes if UTM 32 is not appropriate
    
    with rio.open(viirs, "r") as src: #opens the .tiff file using rasterio library
        
        transform, width, height, = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds) #gets the geotransform, and raster width/height
        
        meta = src.meta.copy() #copies the metadata from the raster - returns a metadata dictionary
        meta.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
                }) #updates the metadata dictionary with the new UTM CRS
    
        with rio.open(viirs.split(".tif")[0]+"_utm.tif", "w", **meta) as dst: #Reprojects the raster to the UTM CRS and writes the new raster file
            reproject(
                    source=rio.band(src, 1),
                    destination=rio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            
            src.close()
                
    with rio.open(viirs.split(".tif")[0]+"_utm.tif", "r") as src:
        
        viirs = src.read(1) #Reads the UTM VIIRS .tiff file and returns it as a numpy array
        
        src.close()
            
    return viirs


def downsample(epsg_code, viirs, name, factor):
    """
    This function downsamples the input raster by a given scale.
    E.g. dimensions/7 downsamples the raster to ~5km per pixel resolution
    :param epsg_code:
    :param viirs:
    :param name:
    :param factor:
    :return:
    """

    dst_crs = epsg_code #EPSG code for an UTM zone
    
    with rio.open(viirs, "r") as src:
        transform, width, height, = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds) #Outputs dimensions and transform for a reprojection

        transform = Affine(transform.a*factor, transform.b, transform.c, 
                           transform.d, transform.e*factor, transform.f) #Updates the Affine transform to match the new pixel size
        
    with rio.open(viirs, "r") as ds:
        data = ds.read(1,
                out_shape = (int(ds.height/factor), int(ds.width/factor)),
                resampling = rio.enums.Resampling.average) #Update the raster extent to match the new pixel size
        
        with rio.open(name, "w", driver = "GTiff", 
                      height = int(ds.height/factor), width = int(ds.width/factor),
                      count = 1, dtype = rio.dtypes.float32,
                      crs = dst_crs, transform = transform) as dst:
        
            dst.write(data, 1) #Write the new array to a GeoTiff file
        
        
    with rio.open(name, "r") as ds:
    
        data = ds.read(1) #Return the new downsampled raster as an array
    
    return data


def compute_kernels(tiff):
    """
    This function uses a UTM VIIRS .tiff file to compute a circular distance kernel.
    :param tiff:
    :return:
    """

    with rio.open(tiff, "r") as src:
        
        getTransform = src.get_transform() #Get the GeoTiff Transform metadata
        image_shape = (src.height, src.width) #Get the raster extent
        pixel_sizeX = getTransform[1] #X Pixel Size - Use print(src.get_transform() to check the output for pixel size, if needed)
        pixel_sizeY = getTransform[-1] #Y Pixel Size
        
        src.close()
    
    kernelX = np.zeros(image_shape) #Initialize an empty array of zeroes matching the shape of the VIIRS input image
    
    """Creates 2-D arrays for X and Y dimensions of the kernel.
        Both arrays are then filled with sequence of numbers from 0 to N and 
        from 0 to -N from the middle row/column (0) of the array, outwards.
        Changing anything here will most likely break the script... Terribly."""
        
    try:
        for i in range(int(kernelX.shape[1]/2)):
            kernelX[:, int(kernelX.shape[1]/2)+i] = i
        
        kernelX[:, -1] = int(kernelX.shape[1]/2)

    except IndexError:
        pass
    
    try:
        for i in range(0, int(kernelX.shape[1]/2), 1):
            kernelX[:, i] = -int(kernelX.shape[1]/2)+i
            
    except IndexError:
        pass
    
    kernelX = kernelX*pixel_sizeX #Multiply by the VIIRS pixel size to match the resolution

    
    kernelY = np.zeros(image_shape) #Initialize an empty array of zeroes matching the shape of the VIIRS input image
    
    try:
        for i in range(int(kernelY.shape[0]/2)):
            kernelY[int(kernelY.shape[0]/2)+i, :] = i
        
    except IndexError:
        pass
    
    try:
        for i in range(0, int(kernelY.shape[0]/2), 1):
            kernelY[i,:] = -int(kernelY.shape[0]/2)+i
            
    except IndexError:
        pass
    
    kernelY = kernelY*pixel_sizeY #Multiply by the VIIRS pixel size to match the resolution
    
    distanceKernel = np.sqrt((kernelX**2+kernelY**2)) #Creates a kernel of distance to the middle of the kernel.

    theta = np.rad2deg(np.arctan2(kernelX,kernelY)) #Creates an array of angles from the middle of the kernel - ranging from -180 to 180 degrees
    theta = np.nan_to_num(theta, 0) #Converts Not a Number entries to 0s (if any)    
    
    return theta, distanceKernel


def calculate_segments(theta, angle, distance_kernel):
    """
    This function calculates the angles for segments in 10 degree range within the distance kernel.
    It uses the theta array to create segments within the distance kernel,
    and computes the Fast Fourier Transform/Inverse Fast Fourier Transform with those segments as kernels.
    :param theta:
    :param angle:
    :param distance_kernel:
    :return:
    """

    segment = np.where((theta > angle) & (theta < angle+10), distance_kernel, 0)
    plt.imshow(np.where((theta > angle) & (theta < angle+10), distance_kernel, 0))
    plt.show()
    
    return segment


def compute_magnitude_spectrum(viirs, dist_kernel):
    """
    This function transforms the input VIIRS image and segment distance kernel from spatial domain,
    into frequency domain using a Fast Fourier Transform.
    :param viirs:
    :param dist_kernel:
    :return:
    """

    f = np.fft.fft2(viirs)
    fshift = np.fft.fftshift(f)
    log_spectrum = 20*np.log(np.abs(fshift))
    
    plt.subplot(121), plt.imshow(viirs, cmap = "gist_gray", vmin=0, vmax=1)
    plt.title("VIIRS image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(log_spectrum, cmap = "gist_gray")
    plt.title("Magnitude Log Transform"), plt.xticks([]), plt.yticks([])
    plt.show()
    
    dist_f = np.fft.fft2(dist_kernel)
    dist_fshift = np.fft.fftshift(dist_f)
    dist_log_spectrum = 20*np.log(np.abs(dist_fshift))
    
    plt.subplot(121), plt.imshow(dist_kernel, cmap = "gist_gray")
    plt.title("Distance Kernel"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dist_log_spectrum, cmap = "gist_gray")
    plt.title("Magnitude Log Transform"), plt.xticks([]), plt.yticks([])
    plt.show()
    
    combinedFshift = fshift*dist_fshift
    plt.imshow(20*np.log(np.abs(combinedFshift)), cmap = "gist_gray")
    plt.title("Combined Magnitudes (Log Transformed)")
    plt.show()
    
    """Uncomment lines starting with 'plt' to show spatial/frequency domain outputs"""
    
    return combinedFshift


def compute_inverse_fourier_transform(combined_magnitude):
    """
    This function computes the inverse FFT and transforms the combined VIIRS and segment kernel,
    from frequency domain back into spatial domain.
    :param combined_magnitude:
    :return:
    """

    
    inverse_f = np.fft.ifft2(combined_magnitude)
    inverse_fshift = np.fft.ifftshift(inverse_f)
    
    return np.abs(inverse_fshift)


def write_raster(source, name, fft_inverse):
    """
    This function writes the combined inverse FFT image (in spatial domain) to a GeoTiff raster
    :param source:
    :param name:
    :param fft_inverse:
    :return:
    """

    with rio.open(source, "r") as src:
        profile = src.profile

    with rio.open(name, "w", driver = "GTiff", 
                  height = fft_inverse.shape[0], width = fft_inverse.shape[1],
                  count = 1, dtype = rio.dtypes.float64,
                  crs = profile["crs"], transform = profile["transform"]) as dst:
        
        dst.write(fft_inverse, 1)
    
    return


def sum_kernels():
    """
    This function computes the values of summed segments to produce a skyglow raster.
    :return:
    """

    segments = glob("./kernel_outputs/*.tif") #Outputs each segment .tif file as a list
    
    def load(name): #Loads each .tif file supplied to the function as a Numpy array
        
        with rio.open(name, "r") as ds:
            array = ds.read(1)
            
        return array
        
    segment_list = [] #Initialises empty list 
    for segment in range(len(segments)):
        segment_list.append(load(segments[segment])) #Appends each segment array to the list
    
    summed_segments = sum(segment_list) #sums all arrays within the list into a single Numpy array
    
    plt.imshow(summed_segments)
    
    return summed_segments


def main():

    return


if __name__ == "__main__":
    main()
