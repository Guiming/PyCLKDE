# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Aug. 10 2021

import os, time, sys, copy
import numpy as np
import json
import matplotlib as mpl
#mpl.use('Qt4Agg')
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import dill
import points, raster, conf


def discrete_cmap(single_class = False, base_cmap=conf.MULTI_CLASS_CMAP, val_label_dict = None):
    '''Create an N-bin discrete colormap from the specified input map'''
    # https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    if not single_class: # multiple classes
        if val_label_dict is None:
            print ('val_label_dict cannot be None')
            sys.exit(1)
        vals = val_label_dict.keys()
        bounds = []
        for val in vals:
            bounds.append(int(val))
        bounds.append(max(bounds) + 1)
        bounds = np.array(bounds)
        bounds.sort()
        N = bounds.size

        if base_cmap is None or type(base_cmap) is str:
            cmap = plt.cm.get_cmap(base_cmap, N)
        else:
            base = plt.cm.get_cmap(base_cmap)
            color_list = base(np.linspace(0.0, 1.0, N))
            cmap_name = base.name + str(N)
            #cmap = base.from_list(cmap_name, color_list, N)
            cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else: # single class
        cmap = mpl.colors.ListedColormap([conf.ONE_CLASS_COLOR])
        norm = None
    return cmap, norm

def extractCovariatesAtPoints(rasters, samples):
    #t0 = time.time()
    ''' extract values at sample points from a list of raster GIS data layers
        it is assumed all rasters have the same spatial extent
    '''
    try:
        t0 = time.time()
        N_PNTS = samples.size
        N_LYRS = len(rasters)

        nd = 0 # num of NoData

        rs, cs = rasters[0].xy2RC_batch(samples.xycoords[:,0], samples.xycoords[:,1])

        vals = np.zeros((N_LYRS, N_PNTS))
        for i in range(N_LYRS):
            if not rasters[i].tileMode:
                tmpvals = rasters[i].getData2D()[rs, cs]
                vals[i, :] = tmpvals
                #print(np.sum(tmpvals==rasters[i].nodatavalue))
            else:
                tmpvals = rasters[i].tileReader.extractByNbrhdRC_batch(rs, cs).reshape((1, -1))
                vals[i, :] = tmpvals
                #print(np.sum(tmpvals==rasters[i].nodatavalue))
        print('+++ extractCovariatesAtPoints took %f s' % (time.time() - t0))
        return vals

    except Exception as e:
        raise

def readEnvDataLayers(rasterdir, rasterfns, pdf=False, tile_mode=False):
    ''' read in environmental data layers
        ascdir: directory containing .asc files
        asciifns: list of ascii file names
        return: list of rasters
    '''
    t0 = time.time()
    rst_list = [] # hold environmental variables
    for rasterfn in rasterfns:  ## presume all covariates are continuous
        rst = None
        if 'landcover' in rasterfn or 'LDC12CLS' in rasterfn: ## Need to modify this to recognize categorical variables
            rst = raster.Raster(msrlevel = conf.MSR_LEVEL_NOMINAL, tile_mode = tile_mode)
        else:
            rst = raster.Raster(msrlevel = conf.MSR_LEVEL_RATIO, tile_mode = tile_mode)
        rst.readRasterGDAL(rasterdir + os.sep + rasterfn, pdf=pdf)
        #print envmap.getData().size
        rst_list.append(rst)
        #print(rst.nodatavalue)
    print('+++ readEnvDataLayers took %f s' % (time.time() - t0))
    return rst_list
