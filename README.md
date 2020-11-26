## Skyglow FFT
This collection of Python scripts can be used to calculate horizontal skyglow maps from VIIRS-DNB images
via Fast Fourier Transform (FFT), as opposed to usual, inefficient pixel-by-pixel methods. 

The dependencies for the scripts are:
1. Numpy
2. Rasterio
3. Matplotlib
4. Tqdm

The main script is "calculate_skyglow.py", with two utility scripts supporting the calculations.
These are "skyglow_utils.py" and "visualisation_utils.py". 
In order to make the implementation somewhat memory efficient, the raster preprocessing is done
prior to calculating skyglow via "raster_preprocessing.py" script.

To produce a horizontal skyglow map from a VIIRS-DNB image, run the calculate_skyglow.py script,
with a given VIIRS-DNB raster path supplied as a command line argument.
