# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Oct. 23 2021
### this is a configuration file specifying various parameters for running PyCLKDE (e.g., PyCLKDE_main.py)

import sys, os, platform
print('Python version: %s' % sys.version)
print('platform: %s' % platform.node())

DOPLOT = True
BLOCK_PLOT = False
SAVEFIG = True

### Options for ploting raster and/or points
LABEL_PNTS = False #label points with id
# cmap options: http://matplotlib.org/examples/color/colormaps_reference.html
ONE_CLASS_COLOR = 'lightgray'
TWO_CLASS_CMAP = 'Set1_r'
MULTI_CLASS_CMAP = 'rainbow'
CONTINUOUS_CMAP = 'RdBu' #'gray' #
PNT_RADIUS = 10
PNT_COLOR = 'black'
PNT_ALPHA = 0.8
RST_ALPHA = 0.5

## if None, will be determined automatically in gdalwrapper (won't be optimal)
## and used for tiled reading of *ALL* raster layers by default
## it's highly recommended to set the values here globally
TILE_XSIZE = 1024*4 # BlockXsize of the GEOTIFF
TILE_YSIZE = 1024*4 # multiple of Blocksize
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow

MASK_ON_FLY = False # if covariates were not masked to a common nodata area, set this to True

#### debug mode
DEBUG_FLAG = False

### parameters for kernel density estimation
N_INTERVALS = 500 # number of intervals over the range of covariate values at which to estimate pdfs
N_HIST_BINS = 50 # number of histogram bins to produce frequency distributions (for plotting purpose only)

global OPENCL_KDE
OPENCL_KDE = True # if False, uses native python implementation. Otherwise, uses (py)opencl implementation

## bandwith option for estimating backgorund pdf and occurrence pdf
global BW_OPTION_BKG
BW_OPTION_BKG = 0
global BW_OPTION_OCCR
BW_OPTION_OCCR = 0


#### measurement level of environmental covariates
MSR_LEVELS = {'0':'nominal', '1':'ordinal', '2':'interval', '3':'ratio', '4':'count'}
MSR_LEVEL_NOMINAL = MSR_LEVELS['0']
MSR_LEVEL_ORDINAL = MSR_LEVELS['1']
MSR_LEVEL_INTERVAL = MSR_LEVELS['2']
MSR_LEVEL_RATIO = MSR_LEVELS['3']
MSR_LEVEL_COUNT = MSR_LEVELS['4']
NOMINAL_KEYWORD_IN_FN = ['geo', 'geology', 'landcover']

### OpenCL platform and device specification (change according to the outputs from running pyopencl_test.py)
global OPENCL_CONFIG
OPENCL_CONFIG = {'Platform': 'Intel(R) OpenCL', 'Device':'Intel(R) Core(TM) i7-4510U CPU @ 2.00GHz'}
#OPENCL_CONFIG = {'Platform': 'AMD Accelerated Parallel Processing', 'Device':'Iceland'}

#### path to .c file containing opencl kernel functions
KDE_KERNEL_FN = os.path.dirname(os.path.realpath(__file__)) + os.sep +'kde_kernel.c'
