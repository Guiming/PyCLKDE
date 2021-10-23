# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Oct. 23 2021

import numpy as np
import numpy.ma as ma
import os, time, sys, copy, platform, json
#import matplotlib as mpl
#mpl.use('Qt4Agg') # https://stackoverflow.com/questions/53529044/importerror-no-module-named-pyqt5
import matplotlib.pyplot as plt
from matplotlib import gridspec

from osgeo import gdal, gdalconst, osr
#sys.path.insert(0, 'utility')
import conf, util, kde, gdalwrapper, points

class Raster:
    '''
    this is a class representing a raster GIS data layer.
    data can be read as a whole (default mode),
    or in a tile by tile fashion if there is not enough memory to hold the whole layer (along with other covariate layers)
    '''

    tileMode = None # read in raster in tile mode? default is False, i.e., read in whole layer
    tileReader = None # gdal raster io wrapper class for reading raster (either as a whole layer, or in tiles)
    tile_xoff = None # postion parameter of the current tile
    tile_yoff = None
    tile_xsize = None
    tile_ysize = None

    uvals = None # unique values, only for categorical variables

    tileWriter = None # gdal raster io wrapper class for writing raster (either as a whole layer, or in tiles)

    __data2D = None # 2D array holding data values, including NoData values.
                    # it can be of the whole layer or just the current tile read in.
                    # similarly applies to all other variables

    __data1D = None # serialized (row-wise) 1D array holding non-NoData values
    __prjString = None # projection
    __sigTimestampe = 0 #signature timestamp, ms
    __measurement_level = None # nominal, ordinal, interval, ratio, count
    msrInt = None # 0, 1, 2, 3, 4
    filename = None

    ## the variables below need the full data layer to compute
    density = None # probability density functoin (pdf) estimated based on all (or a random subset of) cell values (no data excluded of course)
    density_sample = None # pdf estimated based on cell values at sample points (unweighted)
    density_sample_weighted = None # pdf of weighted sample points

    min = None # global min cell value
    max = None # global max cell value
    mean = None # global mean cell value
    std = None # global std of cell values

    ### timing variables
    time_acc_reads = 0.0 # time spent on reads
    time_acc_writes = 0.0 # time spent on writes
    time_acc_kde_bkg = 0.0 # time spent on estimating background pdf
    time_acc_kde_smpl = 0.0 # time spent on estimating sample pdf

    __reader_initialized = False
    __writer_initialized = False

    def __init__(self, msrlevel = conf.MSR_LEVEL_RATIO, tile_mode = False):
        self.__sigTimestampe = int(time.time()*1000) # in ms
        self.__measurement_level = msrlevel
        self.filename = "NA"
        self.tileMode = tile_mode;

        for key in conf.MSR_LEVELS:
            if conf.MSR_LEVELS[key] == msrlevel:
                self.msrInt = int(key)

    def __serialize2Dto1D(self):
        ''' private member function
            serialize raster data from 2d [including NoData values] to 1d [excluding NoData values]
        '''
        try:
            if self.__data2D is not None:
                tmp = self.__data2D.flatten()
                self.__data1D = tmp[tmp != self.nodatavalue]
                #print(tmp)
                #print("!!!", self.__data1D.shape)

        except Exception as e:
            raise

    def readRasterGDAL(self, fn = None, pdf = False, tile_xsize = conf.TILE_XSIZE, tile_ysize = conf.TILE_YSIZE):
        ''' Use gdalwrapper for reading raster
        '''
        if not self.__reader_initialized:
            if fn is None:
                print("***Warning - please provide a file name to read")
                exit(1)
            self.filenamefull = fn
            self.filename = os.path.basename(fn)
        if not self.tileMode:
            #print('started reading %s' % self.filename)
            pass

        self.tile_xsize = tile_xsize
        self.tile_ysize = tile_ysize

        t0 = time.time()
        if not self.__reader_initialized:
            self.tileReader = gdalwrapper.tiledRasterReader(fn, xsize = self.tile_xsize, ysize = self.tile_ysize)
            if conf.TILE_XSIZE is None:
                conf.TILE_XSIZE = self.tileReader.xsize
            if conf.TILE_YSIZE is None:
                conf.TILE_YSIZE = self.tileReader.ysize

            self.__prjString = self.tileReader.projection
            self.geotransform = self.tileReader.geotransform
            self.xllcorner = self.tileReader.geotransform[0]
            self.cellsize = self.tileReader.geotransform[1]

            self.min = self.tileReader.statistics[0][0]
            self.max = self.tileReader.statistics[0][1]
            self.mean = self.tileReader.statistics[0][2]
            self.std = self.tileReader.statistics[0][3]

            self.nodatavalue = self.tileReader.nodata
            self.nrows = self.tileReader.nrows
            self.ncols = self.tileReader.ncols

            self.yllcorner = self.tileReader.geotransform[3] + self.tileReader.geotransform[5] * self.nrows

            self.__reader_initialized = True

        ## read in data only if tileMode is False
        if not self.tileMode:
            tx = time.time()
            self.__data2D = self.tileReader.readWholeRaster()
            self.time_acc_reads += (time.time() - tx) * 1000.0
            #print(self.__data2D.shape)
            self.__serialize2Dto1D()
            if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                self.__retrieveUniquVals()
            if pdf:
                self.computePopulationDistribution()
        else:

            #self.__data2D = self.tileReader.readNextTile()[0]
            tx = time.time()
            data, self.tile_xoff, self.tile_yoff, self.tile_xsize, self.tile_ysize = self.tileReader.readNextTile()
            self.time_acc_reads += (time.time() - tx) * 1000.0
            if data is not None:
                self.__data2D = np.copy(data)
            else:
                self.__data2D = None
            print('started reading %s at tile xoff = %d yoff = %d xsize = %d ysize = %d' % (self.filename, self.tile_xoff, self.tile_yoff, self.tile_xsize, self.tile_ysize))
            ##print("...reading tile xoff = %d yoff = %d xsize = %d ysize = %d" % (self.tile_xoff, self.tile_yoff, self.tile_xsize, self.tile_ysize))
            self.__serialize2Dto1D()

        print ('...reading took %f s' % (time.time()-t0))


    def resetTileReader(self):
        if self.__reader_initialized:
            self.tileReader.reset()
            self.tile_xoff = 0
            self.tile_yoff = 0
            self.tile_xsize = conf.TILE_XSIZE
            self.tile_ysize = conf.TILE_YSIZE
            #self.time_acc_reads = 0
            #sefl.time_acc_writes = 0

    def getData(self):
        ''' return a deep copy of the serialized 1d data
        '''
        if self.tileMode:
            print("***Warning - getData() returns data for only a tile")
        if self.__data1D is not None: return np.copy(self.__data1D)
        else: return None

    def getData2D(self):
        ''' return a deep copy of the 2d data
        '''
        if self.tileMode:
            print("***Warning - getData2D() returns data for only a tile")
        if self.__data2D is not None: return np.copy(self.__data2D)
        else: return None

    def getValByRC(self, r, c):
        ''' return value by row, col
        '''
        if not self.tileMode:
            return self.__data2D[r][c]
        else:
            return self.tileReader.extractByRC(c, r)

    def getValByXY(self, x, y):
        ''' return value by x, y
        '''
        if not self.tileMode:
            r, c = xy2RC(x, y)
            return self.__data2D[r][c]
        else:
            return self.tileReader.extractByXY(x, y)

    def computePopulationDistribution(self, points_bkg = None, n_points = None):
        ''' compute frequency distributions histogram for NOMINAL/ORDINAL or pdf for INTERVAL/RATIO
            estimated based on cell values at (1) points_bkg, or (2) at a random subset of the cells, or (3) at all cells
        '''
        #print('computePopulationDistribution() called')
        t0 = time.time()
        xmin = self.min
        xmax = self.max

        if points_bkg is not None: # (1)
            vals = util.extractCovariatesAtPoints([self], points_bkg)[0]
        elif n_points is not None: # (2)
            points_bkg = points.Points()
            points_bkg.generateRandom(n_points, self)
            vals = util.extractCovariatesAtPoints([self], points_bkg)[0]
        else: # (3)
            if not self.tileMode:
                vals = self.__data1D
            else:
                print("***Warning - neither points_bkg nor n_points is provided for computePopulationDistribution() [tileMode=True]. continue with n_points = 5000 random points")

                #sys.exit(1)
                points_bkg = points.Points()
                points_bkg.generateRandom(5000, self)
                vals = util.extractCovariatesAtPoints([self], points_bkg)[0]

        if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:

            if self.uvals is None:
                self.__retrieveUniquVals()
            tx = time.time()
            y, x = np.histogram(vals, range = (xmin-0.5, xmax+0.5), bins = int(xmax-xmin)+1, density = True)
            self.time_acc_kde_bkg += (time.time() - tx) * 1000.0
            #print('y', y)
            #print('x', x)

            for i in range(y.size):
                if int(xmin + i) in self.uvals:
                    if y[i] == 0: y[i] += 0.00001

        else:
            x = np.linspace(xmin, xmax, conf.N_INTERVALS)

            mykde = kde.KDE()
            #print(vals)
            #print(x)
            tx = time.time()
            y = mykde.evaluate(vals, x, bwoption=conf.BW_OPTION_BKG)
            self.time_acc_kde_bkg += (time.time() - tx) * 1000.0
            #print y
        self.density = y
        print('computePopulationDistribution() took %.2f s' % (time.time() - t0))

    def computeSampleDistribution(self, points_list):
        ''' compute sample frequency distributions histogram for NOMINAL/ORDINAL or pdf for INTERVAL/RATIO
        '''
        #print('computeSampleDistribution() called')
        t0 = time.time()

        xmin = self.min
        xmax = self.max
        #print xmin, xmax

        self.density_sample = []
        self.density_sample_weighted = []
        for points in points_list:
            vals = util.extractCovariatesAtPoints([self], points)[0]
            #print(vals)
            if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:

                if self.uvals is None:
                    self.__retrieveUniquVals()

                #print('landcover')
                tx = time.time()
                y1, x = np.histogram(vals, range = (xmin-0.5, xmax+0.5), bins = int(xmax-xmin)+1, density = True)
                self.time_acc_kde_smpl += (time.time() - tx) * 1000.0
                #print np.unique(vals)
                #print x, y1
                if np.std(points.weights) != 0: # unequal weights
                    tx = time.time()
                    y2, x = np.histogram(vals, weights = points.weights, bins = int(xmax-xmin)+1, density = True)
                    self.time_acc_kde_smpl += (time.time() - tx) * 1000.0
                else:
                    y2 = np.copy(y1)
            else:
                x = np.linspace(xmin, xmax, conf.N_INTERVALS)
                mykde = kde.KDE()
                tx = time.time()
                y1 = mykde.evaluate(vals, x, bwoption=conf.BW_OPTION_OCCR)
                self.time_acc_kde_smpl += (time.time() - tx) * 1000.0

                if np.std(points.weights) != 0:  # unequal weights
                    mykde = kde.KDE()
                    tx = time.time()
                    y2 = mykde.evaluate(vals, x, weights=points.weights, bwoption=conf.BW_OPTION_OCCR)
                    self.time_acc_kde_smpl += (time.time() - tx) * 1000.0
                else:
                    y2 = np.copy(y1)

            self.density_sample.append(y1)
            self.density_sample_weighted.append(y2)

            print('computeSampleDistribution() took %.2f s' % (time.time() - t0))

    def updateRasterData(self, data1d):
        ''' update raster data by passing a 1d array, exclusing NoData values
        '''
        if self.tileMode:
            print("***Warning - updateRasterData() only update data for a tile")
            #return
        # check dimension
        dim = np.shape(data1d)
        if not self.tileMode and dim[0] > self.nrows * self.ncols:
            print ('cannot deserialize 1D to 2D, too many data')
            sys.exit(1)
        if self.tileMode and dim[0] > self.tile_xsize * self.tile_ysize:
            print ('cannot deserialize 1D to 2D, too many data')
            sys.exit(1)
        try:
            self.__data2D[self.__data2D != self.nodatavalue] = data1d
            self.__data1D = np.copy(data1d)


            if not self.tileMode:
                self.__computeStatistics()
                if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                    self.__retrieveUniquVals()

            if not self.tileMode and self.density is not None: self.computePopulationDistribution()

        except Exception as e:
            raise

    def updateRasterData2D(self, data2d):
        ''' update raster data by passing a 2d array, including NoData values
        '''
        if self.tileMode:
            print("***Warning - updateRasterData2D() only updates data for a tile")
            #return
        # check dimension
        dim = np.shape(data2d)
        if not self.tileMode and (dim[0] != self.nrows or dim[1] != self.ncols):
            print ('dimension of the 2D array does not match dimension of the raster')
            sys.exit(1)
        if self.tileMode and (dim[0] != self.tile_ysize or dim[1] != self.tile_xsize):
            print ('dimension of the 2D array does not match dimension of the tile')
            sys.exit(1)
        try:
            self.__data2D = np.copy(data2d)
            self.__serialize2Dto1D()

            if not self.tileMode:
                if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                    self.__retrieveUniquVals()
                self.__computeStatistics()

            if not self.tileMode and self.density is not None: self.computePopulationDistribution()

        except Exception as e:
            raise

    def maskBy(self, mask):
        ''' extract by mask, return a new raster
        '''
        if self.tileMode:
            print("***Warning - no maskBy() for tiles")
            return
        try:
            rst = self.copySelf()
            data2D = rst.getData2D()
            data2D[np.where(mask.getData2D() == mask.nodatavalue)] = rst.nodatavalue
            rst.updateRasterData2D(data2D)

            return rst
        except Exception as e:
            raise

    def integrate(self, rst, integration_strategy='mean'):
        '''
        integrate two rasters cell by cell following the specified integration_strategy,
        'minimum', 'mean', or 'maximum'
        '''
        if self.tileMode:
            print("***Warning - no integrate() for tiles")
            return
        if integration_strategy not in ['minimum', 'mean', 'maximum']:
            print ('integration strategy %s not supported' % integration_strategy)
            sys.exit(1)

        ## dimension mismatch test
        if self.ncols != rst.ncols:
            print ('# of columns does not match')
            sys.exit(1)

        if self.nrows != rst.nrows:
            print ('# of rows does not match')
            sys.exit(1)

        if self.xllcorner != rst.xllcorner:
            print ('xllcorner does not match')
            sys.exit(1)

        if self.yllcorner != rst.yllcorner:
            print ('yllcorner does not match')
            sys.exit(1)

        if self.cellsize != rst.cellsize:
            print ('cellsize does not match')
            sys.exit(1)

        ## now do integration
        int_rst = self.copySelf()
        data1 = int_rst.getData()
        data2 = rst.getData()

        if integration_strategy == 'minimum':
            data1 = np.min([data1, data2], axis = 0)

        if integration_strategy == 'mean':
            data1 = np.mean([data1, data2], axis = 0)

        if integration_strategy == 'maximum':
            data1 = np.max([data1, data2], axis = 0)

        int_rst.updateRasterData(data1)

        return int_rst

    def normalize(self):
        ''' linearly stretch values to [0, 1]
        '''
        if self.tileMode:
            print("***Warning - no normalize() for tiles")
            return
        int_rst = self.copySelf()
        data = int_rst.getData()
        max = data.max()
        min = data.min()
        int_rst.updateRasterData((data - min)/(max - min))
        return int_rst

    def __computeStatistics(self):
        if self.tileMode:
            print("***Warning - __computeStatistics() for tiles")
            return
        if self.__data1D.size > 2:
            self.min = self.__data1D.min()
            self.max = self.__data1D.max()
            self.mean = self.__data1D.mean()
            self.std = self.__data1D.std()

    def __retrieveUniquVals(self):
        ''' retrieve a list of unique values, only for categorical variables
        '''
        if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:

            def create_set(data):
                if '2.7' in sys.version:
                    from sets import Set
                    return Set(data)
                else:
                    return set(data)

            vals = create_set(self.getData().astype('int'))

            if self.tileMode:
                self.readRasterGDAL()
                data = self.getData2D()
                while data is not None:

                    vals = vals.union(create_set(self.getData().astype('int')))

                    self.readRasterGDAL()
                    data = self.getData2D()

                self.resetTileReader()
                self.readRasterGDAL()

            vals = list(vals)
            vals.sort()
            self.uvals = vals

        else:
            print("***Warning - Attemping to retrieve unique values for continuous variable. Did nothing")

    def resetTimes(self):
        self.time_acc_reads = 0.0
        self.time_acc_writes = 0.0
        self.time_acc_kde_bkg = 0.0
        self.time_acc_kde_smpl = 0.0

    def getMsrLevel(self):
        ''' return measurement level
        '''
        return self.__measurement_level

    def setMsrLevel(self, msrlevel):
        ''' set measurement level
        '''
        self.__measurement_level = msrlevel

    def copySelf(self):
        ''' deep copy a raster object
        '''
        if self.tileMode:
            print("***Warning - copySelf() for tiles")
            #return
        try:
            raster = Raster()
            raster.ncols = self.ncols
            raster.nrows = self.nrows
            raster.xllcorner = self.xllcorner
            raster.yllcorner = self.yllcorner
            raster.cellsize = self.cellsize
            raster.nodatavalue = self.nodatavalue

            raster.min = self.min
            raster.max = self.max
            raster.mean = self.mean
            raster.std = self.std

            raster.__data2D = np.copy(self.__data2D)
            raster.__data1D = np.copy(self.__data1D)

            raster.uvals = self.uvals
            raster.time_acc_reads = 0.0
            raster.time_acc_writes = 0.0
            raster.time_acc_kde_bkg = 0.0
            raster.time_acc_kde_smpl = 0.0

            raster.filename = self.filename
            raster.filenamefull = self.filenamefull
            if self.__prjString is not None:
                raster.__prjString = self.__prjString #np.copy(self.__prjString)

            ## additiona attributes to copy
            raster.geotransform = np.copy(self.geotransform)
            raster.tileMode = self.tileMode

            raster.__reader_initialized = self.__reader_initialized
            #raster.tileReader = copy.deepcopy(self.tileReader)
            raster.tile_xoff = self.tile_xoff
            raster.tile_yoff = self.tile_yoff
            raster.tile_xsize = self.tile_xsize
            raster.tile_ysize = self.tile_ysize
            if raster.__reader_initialized:
                raster.tileReader = gdalwrapper.tiledRasterReader(self.filenamefull, xsize = raster.tile_xsize, ysize = raster.tile_ysize)
                if conf.TILE_XSIZE is None:
                    conf.TILE_XSIZE = raster.tileReader.xsize
                if conf.TILE_YSIZE is None:
                    conf.TILE_YSIZE = raster.tileReader.ysize

            raster.__writer_initialized = False
            
            return raster
        except Exception as e:
            raise

    def writeRasterGDAL(self, fn = None, tile_data = None, tile_xoff = None, tile_yoff = None):
        '''write raster using gdalwrapper
        '''
        if tile_data is None:
            print ('started writing %s' % fn)
        else:
            print ('started writing %s at tile xoff = %d yoff = %d' % (fn, tile_xoff, tile_yoff))
        t0 = time.time()
        if not self.__writer_initialized:
            # only write geotiff
            if fn is None:
                fn = self.filename + 'out.tif'
            if fn[-4:] != '.tif':
                fn += 'tif'



            self.tileWriter = gdalwrapper.tiledRasterWriter(fn, \
                                          self.nrows, \
                                          self.ncols, \
                                          1, \
                                          self.geotransform, \
                                          self.__prjString, \
                                          self.nodatavalue)
            self.__writer_initialized = True

        if not self.tileMode:
            tx = time.time()
            self.tileWriter.WriteWholeRaster(self.__data2D)
            self.time_acc_writes += (time.time() - tx) * 1000.0
        else:
            if tile_data is not None and tile_data.size > 0:
                tx = time.time()
                self.tileWriter.writeTile(tile_data, tile_xoff, tile_yoff)
                self.time_acc_writes += (time.time() - tx) * 1000.0

        print( '...writing took %d s' % (time.time()-t0))

    def printInfo(self):
        ''' print out basic info.
        '''
        print (self.filename)
        print ('------HEADER----------------------')
        print ('ncols %d' % self.ncols)
        print ('nrows %d' % self.nrows)
        print ('xllcorner %f' % self.xllcorner)
        print ('yllcorner %f' % self.yllcorner)
        print ('cellsize %f' % self.cellsize)
        print ('nodatavalue %f' % self.nodatavalue)

        print ('------STATS----------------------')
        print ('measurement_level %s %d' % (self.__measurement_level, self.msrInt))

        print ('min %f' % self.min)
        print ('max %f' % self.max)
        print ('mean %f' % self.mean)
        print ('std %f' % self.std)

        print('\n')

    def xy2RC(self, x, y):
        ''' convert x, y coordinates to row, col
        '''
        row = self.nrows - 1 - int((y - self.yllcorner) / self.cellsize)
        col = int((x - self.xllcorner) / self.cellsize)

        if (row >= 0 and row < self.nrows) and (col >= 0 and col < self.ncols):
            return row, col
        else:
            if conf.DEBUG_FLAG: print( '(x, y) out of bound')
            return -1, -1

    def xy2RC_batch(self, xs, ys):
        '''convert xs, ys coordinates to rows, cols (batch mode)
        '''
        xs = np.array(xs)
        ys = np.array(ys)

        rows = self.nrows - 1 - ((ys - self.yllcorner) / self.cellsize).astype('int')
        cols = ((xs - self.xllcorner) / self.cellsize).astype('int')

        if (np.sum(rows < 0) == 0 and np.sum(rows >= self.nrows) == 0) and (np.sum(cols < 0) == 0 and np.sum(cols >= self.ncols) == 0):
            return rows, cols
        else:
            if conf.DEBUG_FLAG: print( 'some (x, y) out of bound')
            return -1, -1

    def rc2XY(self, row, col):
        ''' convert row, col to x, y coordinates
        '''
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            if conf.DEBUG_FLAG: print ('(row, col) out of bound')
            return -1, -1
        y = self.yllcorner + (self.nrows - 0.5 - row) * self.cellsize
        x = self.xllcorner + (col + 0.5) * self.cellsize
        return x, y

    def rc2XY_batch(self, rows, cols):
        ''' convert rows, cols to xs, ys coordinates (batch mode)
        '''
        rows = np.array(rows)
        cols = np.array(cols)
        if np.sum(rows < 0) > 0 or np.sum(rows >= self.nrows) > 0 or np.sum(cols < 0) > 0 or np.sum(cols >= self.ncols) > 0:
            if conf.DEBUG_FLAG: print ('some (row, col) out of bound')
            return -1, -1
        ys = self.yllcorner + (self.nrows - 0.5 - rows) * self.cellsize
        xs = self.xllcorner + (cols + 0.5) * self.cellsize
        return xs, ys

    ## enhanced version
    def rc2POS(self, row, col):
        ''' convert row, col to index in the 1d array
        '''
        pos = -1
        if row >= self.nrows or col >= self.ncols or row < 0 or col < 0: # updated on 5/15/2018
            if conf.DEBUG_FLAG: print ('(row, col) out of bound')
            return pos
        if self.__data2D[row][col] == self.nodatavalue: # NoData at (row, col)
            if conf.DEBUG_FLAG: print ('NoData at (row, col)')
            return pos

        pos = np.sum(self.__data2D[0:row]!=self.nodatavalue)
        pos += np.sum(self.__data2D[row][0:col+1]!=self.nodatavalue)
        #for j in range(col + 1): # bug fixed
        #    if self.__data2D[row][j] != self.nodatavalue:
        #        pos += 1
        return pos-1

    def pos2RC(self, pos):
        ''' convert index in the 1d array to row, col
        '''
        idx = -1
        if pos > len(self.__data1D) or pos < 0: # out of bound
            if conf.DEBUG_FLAG: print ('pos out of bound')
            return idx
        for row in range(self.nrows):
            for col in range(self.ncols):
                if self.__data2D[row][col] != self.nodatavalue:
                    idx += 1
                    if idx == pos:
                        return row, col

    def xy2POS(self, x, y):
        ''' convert x, y to index in the 1d array
        '''
        r, c = self.xy2RC(x, y)
        return self.rc2POS(r, c)

    def pos2XY(self, pos):
        ''' convert x, y to index in the 1d array
        '''
        r, c = self.pos2RC(pos)
        return self.rc2XY(r, c)

    def createRaster(self, data, xoff, yoff, geotransform, projection, nodata=-9999.0, filename='out.tif',
                    statistics=None, density=None, density_sample=None):
        '''create raster from a tile read from a larger raster
            relative location is represented by xoff (col), yoff (row)
        '''
        self.tileMode = False
        self.filename = filename
        self.filenamefull = filename
        self.__prjString = projection
        self.geotransform = np.copy(geotransform)

        self.nrows, self.ncols = data.shape
        self.cellsize = geotransform[1]

        self.xllcorner = geotransform[0] + self.cellsize * xoff
        self.yllcorner = geotransform[3] - self.cellsize * (yoff + self.nrows)

        self.nodatavalue = nodata
        self.__data2D = np.copy(data)

        self.__serialize2Dto1D()

        ## this allows setting desired statistics
        if statistics is None:
            self.__computeStatistics()
        else:
            self.min = statistics[0]
            self.max = statistics[1]
            self.mean = statistics[2]
            self.std = statistics[3]

        ## this allows setting desired pdfs
        if density is not None:
            self.density = np.copy(density)
        if density_sample is not None:
            self.density_sample = []
            for dens in density_sample:
                self.density_sample.append(np.copy(dens))

    def plot(self, title = '', cmap=conf.CONTINUOUS_CMAP, points = None, radius = conf.PNT_RADIUS, color = conf.PNT_COLOR, val_label_dict = None, rst_alpha=conf.RST_ALPHA, pnt_alpha=conf.PNT_ALPHA, block = conf.BLOCK_PLOT, hillshade = None, labelpnts = conf.LABEL_PNTS):
        ''' plot raster in a 2d mesh, pass in a figure title
        '''
        if self.tileMode:
            print("***Warning - no plot() for tiles")
            return
        try:
            ## not sure why there are a few nan's in some cases, but this is a quick fix
            N_nan = np.sum(np.isnan(self.__data1D))
            if N_nan > 0:
                print ('WARNING ' + self.filename + ': replace ' + str(N_nan) + ' nans (' + str(int(N_nan*10000.0/self.__data1D.size)/100.0) + '%) for plot. Write ascii before ploting.')
                m = np.nanmedian(self.__data1D)
                inds = np.where(np.isnan(self.__data1D))
                # replace nan with median
                self.__data1D[inds] = m
                self.updateRasterData(self.__data1D)
            ## end

            if title is '' or title is None:
                title = self.filename
            rstAlpha = 1.0
            # data to plot
            y, x = np.mgrid[slice(self.yllcorner, self.yllcorner + self.nrows * self.cellsize, self.cellsize),
                slice(self.xllcorner, self.xllcorner + self.ncols * self.cellsize, self.cellsize)]
            Zm = ma.masked_where(self.__data2D == self.nodatavalue,self.__data2D)

            if hillshade is not None:
                rstAlpha = conf.RST_ALPHA
                data2D = hillshade.getData2D()
                Zh = ma.masked_where(data2D == hillshade.nodatavalue,data2D)

            fig = plt.figure(int(time.time()*1000), figsize = (12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
            ax = fig.add_subplot(gs[0])
            #ax.grid(False)
            # cmap options: http://matplotlib.org/examples/color/colormaps_reference.html RdBu

            unvals = np.unique(self.__data1D)
            valmin = np.min(unvals)
            valmax = np.max(unvals)

            ############### plot raster
            norm = None
            if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:

                if val_label_dict is not None:
                    vals = val_label_dict.keys()
                    keys = []
                    for val in vals:
                        keys.append(int(val))
                    keys = np.array(keys)

                #if ticklabels is not None:
                    ticklabels_sub = []
                    for val in keys:
                        key = str(val)
                        if val in unvals and key in val_label_dict.keys():
                            ticklabels_sub.append(val_label_dict[key])
                        elif key in val_label_dict.keys():
                            ticklabels_sub.append(val_label_dict[key]+' [NA]')
                        else:
                            ticklabels_sub.append('[NA]')
                else:
                    ticklabels_sub = ['study area']
                    val_label_dict = {}
                    for val in unvals:
                        val_label_dict[int(val)] = str(int(val))
                    keys = np.array(val_label_dict.keys())

                    ticklabels_sub = []
                    for val in keys:
                        if val in unvals and val in val_label_dict.keys():
                            ticklabels_sub.append(val_label_dict[val])
                        elif val in val_label_dict.keys():
                            ticklabels_sub.append(val_label_dict[val]+' [NA]')
                        else:
                            ticklabels_sub.append('[NA]')

                #print(type(keys))
                #print(keys)

                if hillshade is not None:
                    tax = ax.pcolormesh(x, np.flipud(y), Zh, cmap='gray', edgecolors = 'None')
                    tax.set_rasterized(True)

                single_class = True
                ncolor = unvals.size
                if ncolor > 1:
                    single_class = False
                cmap, norm = util.discrete_cmap(single_class = single_class, base_cmap=cmap, val_label_dict=val_label_dict)

                if not single_class:
                    valmin = keys.min()
                    valmax = keys.max()

                cax = ax.pcolormesh(x, np.flipud(y), Zm, cmap=cmap, norm = norm, vmin=valmin, vmax=valmax, alpha = rstAlpha, edgecolors = 'None') #gouraud
                cax.set_rasterized(True)

                ticks = np.unique(self.__data1D)
                if not single_class:
                    ticks = keys
                cbar = fig.colorbar(cax, ticks=ticks, drawedges=False)
                cbar.ax.set_yticklabels(ticklabels_sub)  # vertically oriented colorbar

            else: # ratio/interval variables
                if hillshade is not None:
                    ax.pcolormesh(x, np.flipud(y), Zh, cmap='gray', edgecolors = 'None')
                cax = ax.pcolormesh(x, np.flipud(y), Zm, cmap=cmap, norm = norm, vmin=valmin, vmax=valmax, alpha = rstAlpha, edgecolors = 'None')
                cax.set_rasterized(True)
                cbar = fig.colorbar(cax, drawedges=False, ticks=None)
                #cbar.set_alpha(1)
                #cbar.draw_all()
                #cbar.solids.set_edgecolor("face")
                #cbar.solids.set_rasterized(True)

            ############ plot points
            if points is not None:
                ax.scatter(points.xycoords[:,0], points.xycoords[:,1], s = points.weights * radius, color = color, alpha = pnt_alpha)
                title = title + ' (n=' + str(points.size) + '_' + str(int(np.sum(points.weights != 0))) + ')'

                if labelpnts:
                    for i in range(points.size):
                        if points.weights[i] > 0: ax.text(points.xycoords[:,0][i], points.xycoords[:,1][i], str(int(points.ids[i])), fontsize=10, bbox={'facecolor':'k', 'alpha':0.2, 'pad':1})

            ax.set_title(title)
            # set the limits of the plot to the limits of the data
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')

            ############ plot a histogram
            vals = self.__data1D
            ax2 = fig.add_subplot(gs[1])

            if self.__measurement_level == conf.MSR_LEVEL_NOMINAL or self.__measurement_level == conf.MSR_LEVEL_ORDINAL:
                #den, bins, patches = plt.hist(vals, bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1), normed = True)
                #print bins
                #den = den[den!=0]
                den, x = np.histogram(vals, range = (vals.min()-0.5, vals.max()+0.5), bins = int(vals.max()-vals.min())+1, density = True)
                den = den[den!=0]

                if self.density is not None: den = self.density[self.density != 0]

                pos = np.arange(len(np.unique(vals))) + 0.5
                ax2.barh(pos, den, align='center', color = 'black')

                labels = []
                for val in np.unique(vals):
                    labels.append(str(int(val)))

                ax2.set_ylim(pos.min() - 0.4, pos.max() + 0.4)
                ax2.set_yticks(pos)
                ax2.set_yticklabels(labels)

                # set the limits of the plot to the limits of the data
                ax2.set_xlim(0.0, 1.02 * den.max())
                ax2.set_xticks([0, den.max()])

            else:
                den, bins, patches = ax2.hist(vals, color = 'gray', orientation = 'horizontal', bins = conf.N_HIST_BINS, density = True)
                if self.density is not None: ax2.plot(self.density, np.linspace(vals.min(), vals.max(), conf.N_INTERVALS), color = 'black')
                ax2.set_ylim(vals.min(), vals.max())

                # set the limits of the plot to the limits of the data
                xmax = max(den)
                if self.density is not None: xmax = max(max(den), max(self.density))
                ax2.set_xlim(0, 1.05 * xmax)
                ax2.set_xticks([0, xmax])

                if points is not None:
                    self.computeSampleDistribution([points])
                    ax2.plot(self.density_sample[0], np.linspace(vals.min(), vals.max(), conf.N_INTERVALS), color = 'red')

            #ax2.set_ylabel('Value')
            ax2.set_xlabel('Density')

            if block: # if block, show plot right now
                plt.show()
            else:     # otherwise, draw plot, show them all together at the end
                plt.draw()

            if conf.SAVEFIG:
                #print(type(title))
                #print(title)
                plt.savefig('figs' + os.sep + title + '.png', dpi=300)
                plt.close()

        except Exception as e:
            raise
