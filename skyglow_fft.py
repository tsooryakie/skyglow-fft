import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from glob import glob
plt.style.use("seaborn-talk")

def reprojectToUTM(tiff):
    
    """Reproject a VIIRS raster from native-WGS84 to a UTM CRS e.g UTM 32N"""
    
    dst_crs = "EPSG:25830" #destination Coordinate Reference System - refer to EPSG for CRS codes if UTM 32 is not appropriate
    
    with rio.open(tiff, "r") as src: #opens the .tiff file using rasterio library
        
        transform, width, height, = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds) #gets the geotransform, and raster width/height
        
        meta = src.meta.copy() #copies the metadata from the raster - returns a metadata dictionary
        meta.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
                }) #updates the metadata dictionary with the new UTM CRS
    
        with rio.open(tiff.split(".tif")[0]+"_utm.tif", "w", **meta) as dst: #Reprojects the raster to the UTM CRS and writes the new raster file
            reproject(
                    source=rio.band(src, 1),
                    destination=rio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            
            src.close()
                
    with rio.open(tiff.split(".tif")[0]+"_utm.tif", "r") as src:
        
        viirs = src.read(1) #Reads the UTM VIIRS .tiff file and returns it as a numpy array
        
        src.close()
            
    return viirs

brit_viirs = reprojectToUTM("brit_isles_padded.tif")


def downsample(viirs, name, factor):
    
    """Downsamples the input raster by a given scale e.g. dimensions/7 downsample the raster to ~5km per pixel resolution"""
    
    dst_crs = "EPSG:25830" #EPSG code for UTM 32N
    
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

brit_viirs = downsample("brit_isles_padded_utm.tif", name= "brit_isles_padded5km.tif", factor= 7)


def computeKernels(tiff):
    
    """Uses a UTM VIIRS .tiff file to compute a circular distance kernel (and degree/segmented kernel)"""
    
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

theta, distance_kernel = computeKernels("brit_isles_padded5km.tif")

distance_kernel[distance_kernel == 0] = 0.5
distance_kernel[distance_kernel > 412000] = 0
distance_kernel[distance_kernel < -412000] = 0
distance_kernel[distance_kernel < 0] = 0

distance_kernel = 1/np.log((distance_kernel/1000))**4.2844 #Applies the skyglow decay function to the kernel
distance_kernel = np.nan_to_num(distance_kernel, 0)

#plt.imshow(np.log(distance_kernel))
#plt.title("Distance Kernel")
#plt.show()

def calculateSegments(angle):
    
    """Calculates the angles for segments in 10 degree range within the distance kernel.
        Uses the theta (angle) array to create segments within the distance kernel,
        and computes the FFT/iFFT with those segments as a kernel."""

    segment = np.where((theta > angle) & (theta < angle+10), distance_kernel, 0)
    #plt.imshow(np.where((theta > angle) & (theta < angle+10), distance_kernel, 0))
    #plt.show()
    
    return segment

def computeMagnitudeSpectrum(viirs, dist_kernel):
    
    """Transforms the input VIIRS image and segment distance kernel from
            spatial domain into frequency domain using FFT"""
    
    f = np.fft.fft2(viirs)
    fshift = np.fft.fftshift(f)
    log_spectrum = 20*np.log(np.abs(fshift))
    
    #plt.subplot(121), plt.imshow(viirs, cmap = "gist_gray", vmin=0, vmax=1)
    #plt.title("VIIRS image"), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(log_spectrum, cmap = "gist_gray")
    #plt.title("Magnitude Log Transform"), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    dist_f = np.fft.fft2(dist_kernel)
    dist_fshift = np.fft.fftshift(dist_f)
    dist_log_spectrum = 20*np.log(np.abs(dist_fshift))
    
    #plt.subplot(121), plt.imshow(dist_kernel, cmap = "gist_gray")
    #plt.title("Distance Kernel"), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(dist_log_spectrum, cmap = "gist_gray")
    #plt.title("Magnitude Log Transform"), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    combinedFshift = fshift*dist_fshift
    #plt.imshow(20*np.log(np.abs(combinedFshift)), cmap = "gist_gray")
    #plt.title("Combined Magnitudes (Log Transformed)")
    #plt.show()
    
    """Uncomment lines starting with 'plt' to show spatial/frequency domain outputs"""
    
    return combinedFshift


def computeInverseFourierTransform(combined_magnitude):
    
    """Transforms the combined VIIRS and segment kernel image
        from frequency domain back into the spatial domain"""
    
    inverse_f = np.fft.ifft2(combined_magnitude)
    inverse_fshift = np.fft.ifftshift(inverse_f)
    
    return np.abs(inverse_fshift)


def writeRaster(source, name, fft_inverse):
    
    """Writes the combined inverse FFT image (back in spatial domain) to a GeoTiff raster"""
       
    with rio.open(source, "r") as src:
        profile = src.profile
    
        
    with rio.open(name, "w", driver = "GTiff", 
                  height = fft_inverse.shape[0], width = fft_inverse.shape[1],
                  count = 1, dtype = rio.dtypes.float64,
                  crs = profile["crs"], transform = profile["transform"]) as dst:
        
        dst.write(fft_inverse, 1)
    
    return


for angle in range(-180, 180, 10):
    
    segment_kernel = calculateSegments(angle) #Gets a new 10 degree slice of distance kernel each iteration

    combined_fshift = computeMagnitudeSpectrum(brit_viirs, segment_kernel) #Calculates the FFT for a new slice each iteration
    
    inverse = computeInverseFourierTransform(combined_fshift) #Calculates the inverse of the FFT for each slice
    
    writeRaster("brit_isles_padded5km.tif", "kernel_outputs/fft_segment_kerneltry_"+str(angle)+".tif", inverse) #Writes the FFT skyglow for each slice to a GeoTiff


def sumKernels():
    
    """Get the values of summed segments to produce a skyglow raster"""
    
    segments = glob("./kernel_outputs/*.tif")
    
    def load(name):
        
        with rio.open(name, "r") as ds:
            array = ds.read(1)
            
        return array
        
    segment_list = []
    for segment in range(len(segments)):
        segment_list.append(load(segments[segment]))
    
    summed_segments = sum(segment_list)
    
    plt.imshow(summed_segments)
    return summed_segments

summed_segment_values = sumKernels()

writeRaster("brit_isles_padded5km.tif", "kernel_outputs/fft_max_kernel_stack.tif", summed_segment_values)
    