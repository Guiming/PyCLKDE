# Author: Guiming Zhang
# Last update: 8/1/2021
#http://www.paolocorti.net/2012/03/08/gdal_virtual_formats/
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow
#https://gdal.org/development/rfc/rfc26_blockcache.html


from osgeo import gdal, gdalconst
import glob, psutil
import os, sys, time
import random, math
import numpy as np
import conf

class VRTBuilder:
    '''
    '''
    def __init__(self):
        '''
        '''
    def buildVRT(self, srcFilelist, outVrt):
        gdal.UseExceptions()
        try:
            vrt_options = gdal.BuildVRTOptions(separate=True, VRTNodata=-9999)
            gdal.BuildVRT(outVrt, srcFilelist, options=vrt_options)
        except RuntimeError as e:
            print(e)
            sys.exit(1)

class tiledRasterReader:
    '''
    '''
    def __init__(self, srcRasterfile, xoff=0, yoff=0, xsize=None, ysize=None):
        '''
        '''
        gdal.UseExceptions()
        #print('Initializing reader...')
        self.srcRasterfile = srcRasterfile
        try:
            # could override the default max block cache size
            #gdal.SetCacheMax(2**30) # 1 GB
            self.ds = gdal.Open(self.srcRasterfile, gdalconst.GA_ReadOnly)
            #print('self.ds: ', self.ds)
        except RuntimeError as e:
            print(e)
            sys.exit(1)

        self.fileList = [srcRasterfile]

        ## set proper measurement level (e.g., nominal, ordinal, interval, ratio)
        self.measurement_level_ints = []
        for fn in self.fileList:
            # default level of measurement
            msrlevel = conf.MSR_LEVEL_RATIO
            for keyword in conf.NOMINAL_KEYWORD_IN_FN:
                if keyword in fn:
                    msrlevel = conf.MSR_LEVEL_NOMINAL
                    break
            for key in conf.MSR_LEVELS:
                if conf.MSR_LEVELS[key] == msrlevel:
                    self.measurement_level_ints.append(int(key))
                    break
        self.measurement_level_ints = np.array(self.measurement_level_ints)

        try:
            self.nbands = self.ds.RasterCount
            self.nrows = self.ds.RasterYSize
            self.ncols = self.ds.RasterXSize
            self.geotransform = self.ds.GetGeoTransform()
            self.projection = self.ds.GetProjection()
            #print('%s:\n\t%d rows %d columns' % (self.srcRasterfile, self.nrows, self.ncols))

            band = self.ds.GetRasterBand(1)
            self.nodata = band.GetNoDataValue()

            ## each band may have a different nodata value
            nodatas = []
            for b in range(1, self.nbands+1):
                #print('band %d nodata: %.2f' % (b, self.ds.GetRasterBand(b).GetNoDataValue()))
                nodatas.append(self.ds.GetRasterBand(b).GetNoDataValue())
            self.nodatas = np.array(nodatas)

            '''
            for i in range(1, self.nbands + 1):
                b = self.ds.GetRasterBand(1)
                nd = b.GetNoDataValue()
                print('band %d nd %.2f' % (i, nd))
            '''
            self.block_ysize_base = band.GetBlockSize()[1]
            self.block_xsize_base = band.GetBlockSize()[0]
            #print(self.block_xsize_base, self.block_ysize_base)
        except RuntimeError as e:
            print(e)
            sys.exit(1)
        #print('\t%d x %d' % (self.block_xsize_base, self.block_ysize_base))

        self.__N_TilesRead = 0
        self.xoff, self.yoff = xoff, yoff

        if xsize is None:
            self.xsize = self.block_xsize_base
        elif xsize > self.ncols:
            print('***Warningg - tile xsize exceeds RasterXsize %d' % self.ncols)
            self.xsize = self.ncols
            #sys.exit(1)
        else:
            self.xsize = xsize

        if ysize is None:
            self.ysize = self.block_ysize_base
        elif ysize > self.nrows:
            print('***Warning - tile xsize exceeds RasterYsize %d' % self.nrows)
            self.ysize = self.nrows
            #sys.exit(1)
        else:
            self.ysize = ysize


        ## estimated data size (in MB)
        self.estimate_TotalSize_MB = self.estimateTileSize_MB(self.nrows, self.ncols)
        self.estimate_TileSize_MB = self.estimateTileSize_MB(self.xsize, self.ysize)

        # min, max, mean, stddev
        self.statistics = np.zeros((self.nbands, 4))
        try:
            for i in range(self.nbands):
                self.statistics[i] = self.ds.GetRasterBand(i+1).GetStatistics(0, 1)
                #self.statistics[i] = np.array([0, 1, 0, 1])
        except RuntimeError as e:
            print(e)
            sys.exit(1)

        #print('Done initializing reader...')

    def __del__(self):
        self.__close();

    def estimateTileSize_MB(self, xsize=None, ysize=None):
        '''
        '''
        if xsize is None:
            xsize = self.xsize
        if ysize is None:
            ysize = self.ysize
        return np.array([1.0]).astype('float32').nbytes / 1024.0**2 * xsize * ysize  * self.nbands

    def readWholeRaster(self):
        try:
            return self.ds.ReadAsArray(xoff=0, yoff=0, xsize=None, ysize=None)
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    def readNextTile(self, xsize=None, ysize=None):
        ## update xsize and ysize if needed
        ## PLEASE specify xsize, ysize ONLY ONCE (when reading the first tile)
        if xsize is not None: self.xsize = xsize
        if ysize is not None: self.ysize = ysize

        N_BLOCK_X = int(math.ceil(self.ncols*1.0/self.xsize))
        y = int(self.__N_TilesRead / N_BLOCK_X)
        x = self.__N_TilesRead - y * N_BLOCK_X

        self.xoff = min(x * self.xsize, self.ncols)
        xsize = min(self.xsize, self.ncols - self.xoff)

        self.yoff = min(y * self.ysize, self.nrows)
        ysize = min(self.ysize, self.nrows - self.yoff)

        if self.xoff == self.ncols or self.yoff == self.nrows:
            return (None, self.xoff, self.yoff, 0, 0)

        data = None
        try:
            data = self.ds.ReadAsArray(xoff=self.xoff, yoff=self.yoff, xsize=xsize, ysize=ysize)
        except RuntimeError as e:
            print(e)
            sys.exit(1)
        ## nodatavalues
        #if self.nodata < 0:
        #    data[data < self.nodata] = self.nodata
        self.__N_TilesRead += 1
        return (data, self.xoff, self.yoff, xsize, ysize)

    def setNTilesRead(self, N):
        self.__N_TilesRead = N

    def readNextTileOverlap(self, xsize=None, ysize=None, overlap = 2):
        ## update xsize and ysize if needed
        ## PLEASE specify xsize, ysize ONLY ONCE (when reading the first tile)
        if xsize is not None: self.xsize = xsize
        if ysize is not None: self.ysize = ysize

        N_BLOCK_X = int(math.ceil(self.ncols*1.0/self.xsize))
        y = int(self.__N_TilesRead / N_BLOCK_X)
        x = self.__N_TilesRead - y * N_BLOCK_X

        self.xoff = min(x * self.xsize, self.ncols)
        xsize = min(self.xsize, self.ncols - self.xoff)

        self.yoff = min(y * self.ysize, self.nrows)
        ysize = min(self.ysize, self.nrows - self.yoff)

        if self.xoff == self.ncols or self.yoff == self.nrows:
            return (None, self.xoff, self.yoff, 0, 0, -1, -1)

        data = None

        _xoff = max(0, self.xoff - overlap)
        _yoff = max(0, self.yoff - overlap)

        if _xoff == 0:
            _xsize = min(xsize + overlap, self.ncols - self.xoff)
        else:
            _xsize = min(xsize + 2 * overlap, self.ncols - self.xoff)

        if _yoff == 0:
            _ysize = min(ysize + overlap, self.nrows - self.yoff)
        else:
            _ysize = min(ysize + 2 * overlap, self.nrows - self.yoff)

        #print('inside', self.xoff, self.yoff, self.xsize, self.ysize)
        #print('inside', _xoff, _yoff, _xsize, _ysize)
        try:
            data = self.ds.ReadAsArray(xoff=_xoff, yoff=_yoff, xsize=_xsize, ysize=_ysize)
        except RuntimeError as e:
            print(e)
            sys.exit(1)
        #data = self.ds.ReadAsArray(xoff=self.xoff, yoff=self.yoff, xsize=xsize, ysize=ysize)

        ## nodatavalues
        #if self.nodata < 0:
        #    data[data < self.nodata] = self.nodata

        self.__N_TilesRead += 1

        #return (data, _xoff, _yoff, _xsize, _ysize)
        return (data, self.xoff, self.yoff, xsize, ysize, _xoff, _yoff)

    def reset(self):
        ''' reset after reading tiles
        '''
        self.xoff, self.yoff = 0, 0
        self.__N_TilesRead = 0

    def extractByXY(self, x, y, xsize=1, ysize=1):
        ''' Extract raster value by x, y coordinates
        '''
        xoff = int((x - self.geotransform[0]) / self.geotransform[1])
        yoff = int((y - self.geotransform[3]) / self.geotransform[5])
        return self.ds.ReadAsArray(xoff, yoff, xsize, ysize)

    def extractByNbrhdXY(self, centerX, centerY, nbrXsize=1, nbrYsize=1):
        ''' Extract raster value by x, y coordinates
        '''
        xoff = int((x - self.geotransform[0]) / self.geotransform[1])
        yoff = int((y - self.geotransform[3]) / self.geotransform[5])
        return self.ds.ReadAsArray(xoff-int(nbrXsize/2), yoff-int(nbrYsize/2), nbrXsize, nbrYsize)

    def extractByNbrhdXY_batch(self, centerXs, centerYs, nbrXsize=1, nbrYsize=1):
        ''' Extract raster value by x, y coordinates
        '''
        xoffs = ((centerXs - self.geotransform[0]) / self.geotransform[1]).astype(int) - int(nbrXsize/2)
        yoffs = ((centerYs - self.geotransform[3]) / self.geotransform[5]).astype(int) - int(nbrYsize/2)

        data = None
        for xoff, yoff in zip(xoffs, yoffs):
            #print('Extracting NBRHD (%d, %d)' % (xoff, yoff))
            tmp = self.ds.ReadAsArray(xoff.item(), yoff.item(), nbrXsize, nbrYsize)

            #print(tmp.shape)
            tmp = np.expand_dims(tmp, axis=0)
            #print(tmp.shape)
            if data is None:
                data = tmp
            else:
                data = np.concatenate((data, tmp), axis=0)
            #print('data.shape:', data.shape)
        #print(data.shape)
        return data

    def extractByRC(self, c, r, xsize=1, ysize=1):
        '''Extract raster value by row, col
        '''
        return self.ds.ReadAsArray(c, r, xsize, ysize)

    def extractByNbrhdRC(self, centerR, centerC, nbrXsize=1, nbrYsize=1):
        ''' Extract raster value by row, col coordinates
        '''
        return self.ds.ReadAsArray(centerC-int(nbrXsize/2), centerR-int(nbrYsize/2), nbrXsize, nbrYsize)

    def extractByNbrhdRC_batch(self, centerRs, centerCs, nbrXsize=1, nbrYsize=1):
        ''' Extract raster value by row, col coordinates
        '''
        data = None
        for centerC, centerR in zip(centerCs, centerRs):
            #print('Extracting NBRHD (%d, %d)' % (xoff, yoff))
            #tmp = self.ds.ReadAsArray(centerC.item(), centerR.item(), nbrXsize, nbrYsize)
            tmp = self.ds.ReadAsArray(int(centerC), int(centerR), nbrXsize, nbrYsize)

            #print(tmp.shape)
            tmp = np.expand_dims(tmp, axis=0)
            #print(tmp.shape)
            if data is None:
                data = tmp
            else:
                data = np.concatenate((data, tmp), axis=0)
            #print('data.shape:', data.shape)
        #print(data.shape)
        return data

    def __close(self):
        #print("...tiledRasterReader closed")
        self.ds = None

class tiledRasterWriter:
    '''
    '''
    def __init__(self, outRasterfile, nrows, ncols, nbands, geotransform, projection, nodata, dtype='float'):
        '''
        '''
        self.nrows, self.ncols, self.nbands, self.nodata = nrows, ncols, nbands, nodata
        try:
            driver = gdal.GetDriverByName('GTiff')
            if 'int16' in dtype:
                Dtype = gdal.GDT_Int16
            elif 'int32' in dtype:
                Dtype = gdal.GDT_Int32
            else:
                Dtype = gdal.GDT_Float32
            #self.ds = driver.Create(outRasterfile, self.ncols, self.nrows, self.nbands, Dtype, options = [ 'COMPRESS=LZW', 'BIGTIFF=YES' ])
            self.ds = driver.Create(outRasterfile, self.ncols, self.nrows, self.nbands, Dtype, options = [ 'COMPRESS=DEFLATE', 'BIGTIFF=YES' ])
            self.ds.SetGeoTransform(geotransform)
            self.ds.SetProjection(projection)

            self.bands = []
            for i in range(self.nbands):
                band = self.ds.GetRasterBand(i + 1)
                band.SetNoDataValue(self.nodata)
                self.bands.append(band)
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    def __del__(self):
        self.__close();

    def WriteWholeRaster(self, data):
        try:
            if len(data.shape) == 2 and self.nbands == 1:
                self.bands[0].WriteArray(data, xoff=0, yoff=0)
                self.bands[0].FlushCache()
            elif len(data.shape) == 3 and self.nbands > 1 and data.shape[0] == self.nbands:
                for i in range(self.nbands):
                    self.bands[i].WriteArray(data[i], xoff=0, yoff=0)
                    self.bands[i].FlushCache()
            else:
                print('data dimension does not match raster dimension. exiting...')
                sys.exit(1)

            data = None
        except RuntimeError as e:
            print(e)
            sys.exit(1)

    def writeTile(self, data, xoff, yoff):
        try:
            if len(data.shape) == 2 and self.nbands == 1:
                self.bands[0].WriteArray(data, xoff=xoff, yoff=yoff)
                self.bands[0].FlushCache()
            elif len(data.shape) == 3 and self.nbands > 1 and data.shape[0] == self.nbands:
                for i in range(self.nbands):
                    self.bands[i].WriteArray(data[i], xoff=xoff, yoff=yoff)
                    self.bands[i].FlushCache()
            else:
                print('data dimension does not match raster dimension. exiting...')
                sys.exit(1)

            data = None
        except RuntimeError as e:
            print(e)
            sys.exit(1)
    ## Have to call this to write to disc
    def __close(self):
        #print("...tiledRasterWriter closed")
        try:
            for band in self.bands:
                stats = band.GetStatistics(0, 1)
                #SetStatistics(double min, double max, double mean, double stddev)
                band.SetStatistics(stats[0], stats[1], stats[2], stats[3])

            for band in self.bands:
                band = None

            self.ds = None
        except RuntimeError as e:
            print(e)
            sys.exit(1)
