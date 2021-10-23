# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Aug.23 2021

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import gridspec
import csv, time, random, os, sys, math
import random

if '2.7' in sys.version:
    from sets import Set
import conf, raster, util

class Points:
    '''
    this is a class representing generic field samples (e.g., species occurrence)
    each point has x, y coordinates, a attribute (e.g., 1/0 to indicate presence/absence), a weight, and a prediction
    '''
    ids = None # point ids
    xycoords = None # 2D array holding data values, including NoData values
    attributes = None # attribute values at point locations
    weights = None    # weights
    predictions = None
    size = None

    ### Added 09/08/2017, optional
    covariates_at_points = None # to hold evironmental variates values at points
                                # avoid reading complete data layers to save memory (e.g., large study area)

    __header = None
    __sigTimestampe = 0 #signature timestamp, ms

    def __init__(self):
        self.__sigTimestampe = int(time.time()*1000) # in ms

    def readFromCSV(self, csvfn):
        ''' create points from a csv file (without covariates_at_points)
        '''
        try:
            f = open(csvfn, 'r')
            csvreader = csv.reader(f, delimiter=',')

            self.__header = next(csvreader, None)  # skip the headers
            #print self.__header

            data = []
            for row in csvreader:
                data.append(row)
            f.close()

            data = np.asarray(data, 'float')
            #print '....', np.isnan[data]
            #print data

            self.ids = data[:, 0]
            self.xycoords = data[:, 1:3]
            self.attributes = data[:, 3]
            self.weights = data[:, 4]
            self.predictions = data[:, 5]

            ## read in covariates at points, if any
            N_COLS = data.shape[1]
            if N_COLS > 6:
                self.covariates_at_points = data[:,6:N_COLS]

            self.size = self.weights.size

            '''
            print np.shape(self.xycoords)
            print np.shape(self.attributes)
            print np.shape(self.weights)
            '''
        except Exception as e:
            raise

    def generateRandom(self, N, mask, val = None):
        ''' genrate N random points that is within the extent specified by mask (a raster)
            with uniform probability
        '''
        print('generating random points...')
        try:
            xmin = mask.xllcorner
            xlen = mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ylen = mask.cellsize * mask.nrows
            #vals = mask.getData()

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            cnt = 0
            while cnt < N:
                rows = np.random.choice(range(0, mask.nrows), N)
                cols = np.random.choice(range(0, mask.ncols), N)
                #print(rows)
                #print(cols)
                #print("***\n")
                for i in range(N):
                    r = rows[i]
                    c = cols[i]
                    if mask.tileMode:
                        cval = mask.getValByRC(r, c)
                    else:
                        cval = mask.getData2D()[r,c]

                    if cval != mask.nodatavalue:
                        #print("%d pnt %d cval: %f" % (i, cnt, cval))
                        x, y = mask.rc2XY(r, c)
                        self.xycoords[cnt, 0] = x
                        self.xycoords[cnt, 1] = y
                        if val is None:
                            self.attributes[cnt] = cval
                        else:
                            self.attributes[cnt] = val
                        if cnt % 1000 == 0: print("...random point %d/%d" % (cnt, N))
                        cnt += 1
                        if cnt >= N: break
            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    def generateRandom_v2(self, N, masks, val = None):
        ''' genrate N random points that is within the extent specified by mask (a raster)
            with uniform probability
        '''
        print('generating random points...')
        try:

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            #'''
            datas = []
            for mask in masks: datas.append(mask.getData2D())
            cnt = 0
            while cnt < N:
                print(cnt, '/', N)
                r = np.random.choice(range(0, masks[0].nrows), 1)
                c = np.random.choice(range(0, masks[0].ncols), 1)
                x, y = masks[0].rc2XY(r,c)

                FLAG = True
                for mask, data in zip(masks, datas):
                    rx, cx = mask.xy2RC(x,y)
                    if data[rx,cx] == mask.nodatavalue:
                        FLAG = False
                        break

                if not FLAG:
                    continue

                if np.isnan(datas[0][r,c]):
                    continue

                #x, y = masks[0].rc2XY(r,c)
                self.xycoords[cnt, 0] = x
                self.xycoords[cnt, 1] = y
                self.attributes[cnt] = datas[0][r,c]
                if np.isnan(datas[0][r,c]):
                    print(r, c, datas[0][r,c])
                cnt += 1

            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    def generateBiasedRandom(self, mask, seedxy = None, N=10, decay = 1.0, dist_threshold = 0.1):
        ''' genrate N random points that is within the extent specified by mask (a raster)
            with decreasing probability away from the seed point
            TO BE IMPLEMENTED
        '''
        try:
            xmin = mask.xllcorner
            xlen = mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ylen = mask.cellsize * mask.nrows
            vals = mask.getData()

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            ### Added 09/08/2017
            dmax = math.sqrt(xlen**2 + ylen**2)

            idxs = range(0, mask.getData().size)

            idxsA = [] # the key is to determine the selected ids, i.e., pos of pixels
            if seedxy is None: # no seed point provided
                idx_seed = random.sample(idxs, 1)[0]
                x, y = mask.pos2XY(idx_seed)
                seedxy = [x, y]

            cnt = 0
            while cnt < N:
                print('##in while loop - generateBiasedRandom()')
                ### generate 10 * N random points
                delta = dist_threshold * dmax
                xys = np.array([seedxy[0] + (np.random.uniform(size=20*N) - 0.5) * delta, seedxy[1] + (np.random.uniform(size=20*N) - 0.5) * delta]).T
                #print xys.shape, xys[0]
                dists = np.sqrt(np.sum((xys - np.array(seedxy))**2, axis=1))
                dists[dists == 0] = 0.01
                prob = 1.0 / dists**decay
                prob = prob / np.sum(prob)  #
                #print prob

                idxsA = np.random.choice(range(20*N), size=N, replace=False, p=prob)

                for i in range(len(idxsA)):
                    xy_tmp = xys[idxsA[i]]
                    pos = mask.xy2POS(xy_tmp[0], xy_tmp[1])
                    #print 'i = ', i
                    if pos != -1 and cnt < N:
                        self.xycoords[cnt] = xys[idxsA[i]]
                        self.attributes[cnt] = mask.getData()[pos]
                        #print 'cnt', cnt
                        cnt = cnt + 1

            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    def generateStratifiedRandom(self, N, mask):
        ''' genrate N stratified random points that is within the extent specified by mask (a raster)
        '''
        try:
            msrlevel = mask.getMsrLevel()
            if not (msrlevel == conf.MSR_LEVEL_NOMINAL or msrlevel == conf.MSR_LEVEL_ORDINAL):
                print('strata raster must be nominal or ordinal')
                sys.exit(1)

            xmin = mask.xllcorner
            xlen = mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ylen = mask.cellsize * mask.nrows
            vals = mask.getData()
            uvals = np.unique(vals)

            proportions = []
            for val in uvals:
                proportions.append(np.sum(vals==val)*1.0/vals.size)

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            counter = 0
            for i in range(uvals.size):
                Ni = int(N * proportions[i] + 0.5)
                idx = np.where(vals == uvals[i])[0]
                idx_chosen = np.random.choice(idx, size=Ni, replace=True)

                #print uvals[i], len(idx_chosen)

                for j in range(len(idx_chosen)):
                    if counter < N: # in case of round up error
                        x, y = mask.pos2XY(idx_chosen[j])
                        self.xycoords[counter, 0] = x + (random.random() - 0.499) * mask.cellsize
                        self.xycoords[counter, 1] = y + (random.random() - 0.499) * mask.cellsize
                        self.attributes[counter] = vals[idx_chosen[j]]
                        counter += 1
            while counter < N: # in case of round down error
                pos = np.random.choice(range(vals.size), size=1, replace=False)
                x, y = mask.pos2XY(pos)
                self.xycoords[counter, 0] = x + (random.random() - 0.499) * mask.cellsize
                self.xycoords[counter, 1] = y + (random.random() - 0.499) * mask.cellsize
                self.attributes[counter] = vals[pos]
                counter += 1

            print(np.min(self.xycoords, axis = 0))
            #print self.xycoords.max()
            #print counter

            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']

        except Exception as e:
            raise

    def generateRegular(self, grideSize, mask):
        ''' generate regular grid points (with random start)
            within the extent specified by mask (a raster)
        '''
        try:
            xmin = mask.xllcorner
            xmax = xmin + mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ymax = ymin + mask.cellsize * mask.nrows
            vals = mask.getData()

            attributes = []
            xycoords = [[],[]]
            weights = []

            # random starting point
            x0 = xmin + random.random() * (xmax - xmin)
            y0 = ymin + random.random() * (ymax - ymin)

            xs = [] # possible x coords of the sampled points
            x = x0
            while x < xmax:
                xs.append(x)
                x = x + grideSize
            x = x0 - grideSize
            while x > xmin:
                xs.append(x)
                x = x - grideSize

            ys = [] # possible x coords of the sampled points
            y = y0
            while y < ymax:
                ys.append(y)
                y = y + grideSize
            y = y0 - grideSize
            while y > ymin:
                ys.append(y)
                y = y - grideSize

            # combine x, y coords to locate grid points
            for x in xs:
                for y in ys:
                    pos = mask.xy2POS(x, y)
                    if pos != -1: # skip those NoData locations
                        xycoords[0].append(x)
                        xycoords[1].append(y)
                        attributes.append(vals[pos])
                        weights.append(1.0)
            self.xycoords = np.array(xycoords).T
            self.attributes = np.array(attributes)
            self.weights = np.array(weights)
            self.size = self.weights.size
            self.ids = range(0, self.size, 1)
            self.predictions = np.zeros(self.size)

            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    #return random subset
    def RandomSubset(self, N, probs=None):
        ''' return a random subset of points of size N.
            probs includes the selection probabilities
        '''
        pnts = Points()
        pnts.__header = np.copy(self.__header)

        if N > self.size: N = self.size

        if probs is None:
            probs = np.ones(self.size)
        else:
            probs = np.array(probs)
        if probs.size != self.size:
            print('dimension of probs not equal to number of points')
            sys.exit(1)

        #probs = probs ** 2 # **1
        probs = probs ** 1
        if np.sum(probs) != 1.0:
            probs = probs / np.sum(probs)

        idxsA = np.random.choice(range(self.size), size=N, replace=False, p=probs)

        pnts.ids = self.ids[idxsA]
        pnts.xycoords = self.xycoords[idxsA]
        pnts.attributes = self.attributes[idxsA]
        pnts.weights = self.weights[idxsA]
        pnts.predictions = self.predictions[idxsA]
        pnts.size = pnts.ids.size

        ## split covariates_at_points as well, if any
        if self.covariates_at_points is not None:
            pnts.covariates_at_points = self.covariates_at_points[idxsA]

        return pnts

    def updateWeights(self, weights):
        ''' update weights of the points
        '''
        try:
            if np.shape(weights) != np.shape(self.weights):
                print('weights dimension does not match')
                sys.exit(1)
            self.weights = np.copy(weights)
        except Exception as e:
            raise

    def copySelf(self):
        ''' copy points themselves
        '''
        try:
            points = Points()
            points.ids = np.copy(self.ids)
            points.xycoords = np.copy(self.xycoords)
            points.attributes = np.copy(self.attributes)
            points.weights = np.copy(self.weights)
            points.predictions = np.copy(self.predictions)
            points.__header = np.copy(self.__header)
            points.size = len(points.weights)

            ## copy covariates values at points, if any
            if self.covariates_at_points is not None:
                points.covariates_at_points = np.copy(self.covariates_at_points)

            return points

        except Exception as e:
            raise

    def mergePoints(self, aPoints):
        ''' merge with aPoints, return a new Points object
        '''
        try:
            NA = self.size
            NB = aPoints.size
            points = Points()
            points.ids = np.zeros(NA + NB)
            points.xycoords = np.zeros((self.size + aPoints.size, 2))
            points.attributes = np.zeros(NA + NB)
            points.weights = np.zeros(NA + NB)
            points.predictions = np.zeros(NA + NB)
            points.__header = np.copy(self.__header)
            points.size = NA + NB

            points.ids[0:NA] = np.copy(self.ids)
            points.xycoords[0:NA] = np.copy(self.xycoords)
            points.attributes[0:NA] = np.copy(self.attributes)
            points.weights[0:NA] = np.copy(self.weights)
            points.predictions[0:NA] = np.copy(self.predictions)

            points.ids[NA:] = np.copy(aPoints.ids)
            points.xycoords[NA:] = np.copy(aPoints.xycoords)
            points.attributes[NA:] = np.copy(aPoints.attributes)
            points.weights[NA:] = np.copy(aPoints.weights)
            points.predictions[NA:] = np.copy(aPoints.predictions)

            ## copy covariates values at points, if any
            if self.covariates_at_points is not None and aPoints.covariates_at_points is not None:
                aPoints.covariates_at_points.shape[1] = NC
                if self.covariates_at_points.shape[1] == NC:
                    points.covariates_at_points = np.zeros((NA + NB, NC))
                    points.covariates_at_points[0:NA] = np.copy(self.covariates_at_points)
                    points.covariates_at_points[NA:] = np.copy(aPoints.covariates_at_points)

            return points

        except Exception as e:
            raise

    def intersectPoints(self, aPoints):
        ''' intersect two point sets by (location), each Points are free of duplicates
            return a new Points object, the new Points object has the attributes of aPoints
        '''
        try:

            def numbers2string(xy):
                return str(int(xy[0]*100)) + '&' + str(int(xy[1]*100))

            xys_set = Set([])
            for i in range(self.size):
                tmp_xy = numbers2string(self.xycoords[i])
                xys_set.add(tmp_xy)

            xys_a_set = Set([])
            idx_a = []
            for i in range(aPoints.size):
                tmp_xy = numbers2string(aPoints.xycoords[i])
                xys_a_set.add(tmp_xy)
                idx_a.append(tmp_xy)

            common_set = xys_a_set.intersection(xys_set)
            common_ids = []
            for i in range(len(idx_a)):
                if idx_a[i] in common_set:
                    common_ids.append(aPoints.ids[i])

            return aPoints.SelectPointsByIds(common_ids)

        except Exception as e:
            raise

    def randomSplit(self, ratio = [0.5, 0.5]):
        ''' randomly split points into two sets given the number of points raito
            it is the same as random selection
            could use this for random selection
        '''
        splitA = Points()
        splitA.__header = np.copy(self.__header)
        splitB = Points()
        splitB.__header = np.copy(self.__header)

        NA = int(self.size * 1.0 * ratio[0]/sum(ratio))
        idxs = range(self.size)
        idxsA = random.sample(idxs, NA)
        splitA.ids = self.ids[idxsA]
        splitA.xycoords = self.xycoords[idxsA]
        splitA.attributes = self.attributes[idxsA]
        splitA.weights = self.weights[idxsA]
        splitA.predictions = self.predictions[idxsA]
        splitA.size = NA

        NB = self.size - NA
        idxsB = list(set(idxs) - set(idxsA))
        splitB.ids = self.ids[idxsB]
        splitB.xycoords = self.xycoords[idxsB]
        splitB.attributes = self.attributes[idxsB]
        splitB.weights = self.weights[idxsB]
        splitB.predictions = self.predictions[idxsB]
        splitB.size = NB

        ## split covariates_at_points as well, if any
        if self.covariates_at_points is not None:
            splitA.covariates_at_points = self.covariates_at_points[idxsA]
            splitB.covariates_at_points = self.covariates_at_points[idxsB]

        return splitA, splitB

    def randomBiasedSelection(self, seed_id = None, N=10, decay=1.0):
        ''' randomly select a point as seed, then select other points with decreasing probability
            away from this seed point (distance decay factor), until get N points in total
            decay - distance decay factor, similar to that of IDW
        '''
        splitA = Points() # to hold the selected points
        splitA.__header = np.copy(self.__header)

        idxs = range(self.size)

        if seed_id is None: # no seed point provided
            idx_seed = random.sample(idxs, 1)[0]
        else:
            idx_seed = self.ids.tolist().index(seed_id)

        xy_seed = self.xycoords[idx_seed]
        #print (self.xycoords - xy_seed)**2
        dists = np.sqrt(np.sum((self.xycoords - xy_seed)**2, axis=1))
        dists[dists == 0] = 0.01
        prob = 1.0 / dists**decay
        prob = prob / np.sum(prob)  #
        #print prob
        #print dists
        idxsA = np.random.choice(idxs, size=N, replace=False, p=prob)

        splitA.ids = self.ids[idxsA]
        splitA.xycoords = self.xycoords[idxsA]
        splitA.attributes = self.attributes[idxsA]
        splitA.weights = self.weights[idxsA]
        splitA.predictions = self.predictions[idxsA]
        splitA.size = N

        ## split covariates_at_points as well, if any
        if self.covariates_at_points is not None:
            splitA.covariates_at_points = self.covariates_at_points[idxsA]

        return splitA

    #return non-zero weight points
    def nonZeroWeightPoints(self):
        ''' return non-zero weight points
        '''
        pnts = Points()
        pnts.__header = np.copy(self.__header)

        idxs = np.array(range(self.size))
        idxsA = idxs[self.weights != 0.0]

        pnts.ids = self.ids[idxsA]
        pnts.xycoords = self.xycoords[idxsA]
        pnts.attributes = self.attributes[idxsA]
        pnts.weights = self.weights[idxsA]
        pnts.predictions = self.predictions[idxsA]
        pnts.size = pnts.ids.size

        ## split covariates_at_points as well, if any
        if self.covariates_at_points is not None:
            pnts.covariates_at_points = self.covariates_at_points[idxsA]

        return pnts

    #return non-zero weight points
    def SelectPointsByIds(self, ids):
        ''' return subset of points by ids
        '''
        pnts = Points()
        pnts.__header = np.copy(self.__header)

        idxs = []
        for id in ids:
            idxs.append(np.where(self.ids==id)[0][0])
        idxsA = np.array(idxs)

        pnts.ids = self.ids[idxsA]
        pnts.xycoords = self.xycoords[idxsA]
        pnts.attributes = self.attributes[idxsA]
        pnts.weights = self.weights[idxsA]
        pnts.predictions = self.predictions[idxsA]
        pnts.size = pnts.ids.size

        ## split covariates_at_points as well, if any
        if self.covariates_at_points is not None:
            pnts.covariates_at_points = self.covariates_at_points[idxsA]

        return pnts

    def ExcludePointsByIds(self, ids):
        ''' return subset of points, with other points in ids removed
        '''
        pnts = Points()
        pnts.__header = np.copy(self.__header)

        idxs = list(range(self.size))
        #print("ids", ids)
        for id in ids:
            #print("id", id)
            idxs.remove(np.where(self.ids==id)[0][0])
        idxsA = np.array(idxs)

        if idxsA.size == 0:
            print ('no points within the mask')
            pnts.ids = None
            pnts.xycoords = None
            pnts.attributes = None
            pnts.weights = None
            pnts.predictions = None
            pnts.size = 0
            if self.covariates_at_points is not None:
                pnts.covariates_at_points = None
        else:
            pnts.ids = self.ids[idxsA]
            pnts.xycoords = self.xycoords[idxsA]
            pnts.attributes = self.attributes[idxsA]
            pnts.weights = self.weights[idxsA]
            pnts.predictions = self.predictions[idxsA]
            pnts.size = pnts.ids.size

            ## split covariates_at_points as well, if any
            if self.covariates_at_points is not None:
                pnts.covariates_at_points = self.covariates_at_points[idxsA]

        return pnts

    def maskBy(self, mask):
        ''' return subset of points that are within the raster mask layer
        '''
        if mask.tileMode:
            print("***Warning - no maskBy() on raster tile")
            return
        try:
            ids = []
            for i in range(self.size):
                if mask.xy2POS(self.xycoords[i][0], self.xycoords[i][1]) == -1:
                    ids.append(self.ids[i])
            return self.ExcludePointsByIds(ids)
        except Exception as e:
            raise

    def eliminateDuplicates(self):
        ''' eliminate point with the same x, y coordinates
        '''
        def numbers2string(xy):
            return str(xy[0]) + '&' + str(xy[1])
        ex_id = []
        if '2.7' in sys.version: xys = Set([])
        else: xys = set([])
        for i in range(self.size):
            tmp_id = self.ids[i]
            tmp_xy = numbers2string(self.xycoords[i])
            if tmp_xy not in xys:
                xys.add(tmp_xy)
            else:
                ex_id.append(tmp_id)
        return self.ExcludePointsByIds(ex_id)

    def writeCSV(self, csvfn, var_names = None):
        ''' write points to csv file
        '''
        try:
            f = open(csvfn, 'wb')
            writer = csv.writer(f)
            if self.covariates_at_points is not None and len(self.__header) == 6:
                if var_names is None:
                    for i in range(self.covariates_at_points.shape[1]):
                        self.__header.append('VAR'+str(i+1))
                else:
                    for i in range(self.covariates_at_points.shape[1]):
                        self.__header.append(var_names[i])
            writer.writerow(self.__header)
            data = np.concatenate((np.array([self.ids]).T, self.xycoords), axis = 1)
            data = np.concatenate((data, np.array([self.attributes]).T), axis = 1)
            data = np.concatenate((data, np.array([self.weights]).T), axis = 1)
            data = np.concatenate((data, np.array([self.predictions]).T), axis = 1)
            #print data.shape

            ## write covariates_at_points as well, if any
            if self.covariates_at_points is not None:
                data = np.concatenate((data, self.covariates_at_points), axis = 1)
            #print data.shape

            writer.writerows(data)
            f.close()

        except Exception as e:
            raise

    def writeCSVMXT(self, csvfn, species='species', varnames = None):
        ''' write points to csv file in MAXENT format
        '''
        try:
            f = open(csvfn, 'wb')
            writer = csv.writer(f)

            header = ['species', 'x', 'y']
            if self.covariates_at_points is not None:
                if varnames is not None:
                    if len(varnames) >= self.covariates_at_points.shape[1]:
                        for i in range(self.covariates_at_points.shape[1]):
                            header.append(varnames[i])
                    else:
                        print ('not enough iterms in varnames')
                elif len(self.__header) == (6+self.covariates_at_points.shape[1]):
                    for i in range(self.covariates_at_points.shape[1]):
                        header.append(self.__header[6 + i])
                else:
                    for i in range(self.covariates_at_points.shape[1]):
                        header.append('VAR'+str(i))

            writer.writerow(header)
            sp = np.ones((self.size,1)).astype('str')
            #print self.size, sp.shape
            sp[:] = species
            #print sp
            data = np.concatenate((sp, self.xycoords), axis = 1)
            #data = np.concatenate((np.array([self.ids]).T, self.xycoords), axis = 1)
            #data = np.concatenate((data, np.array([self.attributes]).T), axis = 1)
            #data = np.concatenate((data, np.array([self.weights]).T), axis = 1)
            #data = np.concatenate((data, np.array([self.predictions]).T), axis = 1)
            #print data.shape

            ## write covariates_at_points as well, if any
            if self.covariates_at_points is not None:
                data = np.concatenate((data, self.covariates_at_points), axis = 1)
            #print data.shape

            ## weight > 1
            ids = np.array(range(0,self.size))
            ids = ids[np.where(self.weights > 1)]
            #print data.shape
            #print data[0,:]
            for id in ids:
                for i in range(int(self.weights[id]-0.5)):
                    data = np.concatenate((data, np.reshape(data[id,:], (1, data.shape[1]))), axis = 0)

            writer.writerows(data)
            f.close()

        except Exception as e:
            raise

    def plot(self, title = '', radius = conf.PNT_RADIUS, color = conf.PNT_COLOR, alpha=conf.PNT_ALPHA, block = conf.BLOCK_PLOT, labelpnts = conf.LABEL_PNTS):
        ''' scatter plot the points, size proportional to weights, no background
        '''
        try:
            x = self.xycoords[:,0]
            y = self.xycoords[:,1]

            fig = plt.figure(int(time.time()*1000))
            ax = fig.add_subplot(111)

            ax.scatter(x, y, s = self.weights * radius, color = color, alpha = alpha, label = 'occurrences')
            #title = title + ' (n=' + str(self.size) + '_' + str(int(np.sum(self.weights != 0))) + ')'
            title = title + ' (n=' + str(self.size) + ')'

            if labelpnts:
                for i in range(self.size):
                    if self.weights[i] > 0: ax.text(x[i], y[i], str(int(self.ids[i])), fontsize=10, bbox={'facecolor':'k', 'alpha':0.2, 'pad':1})

            ax.set_title(title)
            # set the limits of the plot to the limits of the data
            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()

            ax.axis([xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin), ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)])
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')

            if block: # if block, show plot right now
                plt.show()
            else:     # otherwise, draw plot, show them all together at the end
                plt.draw()

            if conf.SAVEFIG:
                plt.savefig('figs' + os.sep + title + '.png', dpi = 300)
                plt.close()

        except Exception as e:
            raise

    def plot_background(self, raster, hillshade, title = '', cmap=conf.CONTINUOUS_CMAP, radius = conf.PNT_RADIUS, color = conf.PNT_COLOR, val_label_dict = None, rst_alpha=conf.RST_ALPHA, pnt_alpha=conf.PNT_ALPHA, block = conf.BLOCK_PLOT, labelpnts = conf.LABEL_PNTS):
        ''' plot points using a raster layer as background
        '''
        if raster.tileMode or hillshade.tileMode:
            print("***Warning - no plot_background() on raster tile")
            return
        try:
            fig = plt.figure(int(time.time()*1000))
            ax = fig.add_subplot(111)

            rstAlpha = 1.0

            if hillshade is not None:
                y, x = np.mgrid[slice(hillshade.yllcorner, hillshade.yllcorner + hillshade.nrows * hillshade.cellsize, hillshade.cellsize),
                    slice(hillshade.xllcorner, hillshade.xllcorner + hillshade.ncols * hillshade.cellsize, hillshade.cellsize)]

                rstAlpha = conf.RST_ALPHA
                data2D = hillshade.getData2D()
                Zh = ma.masked_where(data2D == hillshade.nodatavalue,data2D)

                ax.pcolormesh(x, np.flipud(y), Zh, cmap='gray', edgecolors = 'None')


            #if raster is not None: raster cannot be None
            y, x = np.mgrid[slice(raster.yllcorner, raster.yllcorner + raster.nrows * raster.cellsize, raster.cellsize),
                slice(raster.xllcorner, raster.xllcorner + raster.ncols * raster.cellsize, raster.cellsize)]
            Zm = ma.masked_where(raster.getData2D() == raster.nodatavalue, raster.getData2D())

            unvals = np.unique(raster.getData())
            valmin = np.min(unvals)
            valmax = np.max(unvals)

            norm = None

            if raster.getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:

                if val_label_dict is not None:
                    vals = val_label_dict.keys()
                    keys = []
                    for val in vals:
                        keys.append(int(val))
                    keys = np.array(keys)

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

                ticks = np.unique(raster.getData())
                if not single_class:
                    ticks = keys
                cbar = fig.colorbar(cax, ticks=ticks, drawedges=False)
                cbar.ax.set_yticklabels(ticklabels_sub)  # vertically oriented colorbar

            else: # ratio/interval variables
                cax = ax.pcolormesh(x, np.flipud(y), Zm, cmap=cmap, norm = norm, vmin=valmin, vmax=valmax, alpha = rstAlpha, edgecolors = 'None')
                cax.set_rasterized(True)
                cbar = fig.colorbar(cax, drawedges=False, ticks=None)

            ############ plot points
            ax.scatter(self.xycoords[:,0], self.xycoords[:,1], s = self.weights * radius, color = color, alpha = pnt_alpha)
            #title = title + ' (n=' + str(self.size) + '_' + str(int(np.sum(self.weights != 0))) + ')'
            title = title + ' (n=' + str(self.size) + ')'

            if labelpnts:
                for i in range(self.size):
                    if self.weights[i] > 0: ax.text(self.xycoords[:,0][i], self.xycoords[:,1][i], str(int(self.ids[i])), fontsize=10, bbox={'facecolor':'k', 'alpha':0.2, 'pad':1})

            ax.set_title(title)
            # set the limits of the plot to the limits of the data
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')

            if block: # if block, show plot right now
                plt.show()
            else:     # otherwise, draw plot, show them all together at the end
                plt.draw()

            if conf.SAVEFIG:
                plt.savefig('figs' + os.sep + title + '.png', dpi=300)
                plt.close()

        except Exception as e:
            raise

    def plot_density(self, raster, hillshade, title = '', cmap=conf.CONTINUOUS_CMAP, radius = conf.PNT_RADIUS, color = conf.PNT_COLOR, val_label_dict = None, rst_alpha=conf.RST_ALPHA, pnt_alpha=conf.PNT_ALPHA, block = conf.BLOCK_PLOT, labelpnts = conf.LABEL_PNTS):
        ''' plot points using a raster layer as background (left)
            and density curves (right)
        '''
        if raster.tileMode or hillshade.tileMode:
            print("***Warning - no plot_density() on raster tile")
            return
        try:
            fig = plt.figure(int(time.time()*1000), figsize = (12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
            ax = fig.add_subplot(gs[0])

            rstAlpha = 1.0

            if hillshade is not None:

                y, x = np.mgrid[slice(hillshade.yllcorner, hillshade.yllcorner + hillshade.nrows * hillshade.cellsize, hillshade.cellsize),
                    slice(hillshade.xllcorner, hillshade.xllcorner + hillshade.ncols * hillshade.cellsize, hillshade.cellsize)]

                rstAlpha = conf.RST_ALPHA
                data2D = hillshade.getData2D()
                Zh = ma.masked_where(data2D == hillshade.nodatavalue,data2D)

                ax.pcolormesh(x, np.flipud(y), Zh, cmap='gray', edgecolors = 'None')


            # raster cannot be none
            y, x = np.mgrid[slice(raster.yllcorner, raster.yllcorner + raster.nrows * raster.cellsize, raster.cellsize),
                slice(raster.xllcorner, raster.xllcorner + raster.ncols * raster.cellsize, raster.cellsize)]
            Zm = ma.masked_where(raster.getData2D() == raster.nodatavalue, raster.getData2D())

            unvals = np.unique(raster.getData())
            valmin = np.min(unvals)
            valmax = np.max(unvals)

            norm = None

            if raster.getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                if val_label_dict is not None:
                    vals = val_label_dict.keys()
                    keys = []
                    for val in vals:
                        keys.append(int(val))
                    keys = np.array(keys)

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

                ticks = np.unique(raster.getData())
                if not single_class:
                    ticks = keys
                cbar = fig.colorbar(cax, ticks=ticks, drawedges=False)
                cbar.ax.set_yticklabels(ticklabels_sub)  # vertically oriented colorbar

            else: # ratio/interval variables
                cax = ax.pcolormesh(x, np.flipud(y), Zm, cmap=cmap, norm = norm, vmin=valmin, vmax=valmax, alpha = rstAlpha, edgecolors = 'None')
                cax.set_rasterized(True)
                cbar = fig.colorbar(cax, drawedges=False, ticks=None)

            ############ plot points
            ax.scatter(self.xycoords[:,0], self.xycoords[:,1], s = self.weights * radius, color = color, alpha = pnt_alpha)
            if title == '':
                title = raster.filename
            #title = title + ' (n=' + str(self.size) + '_' + str(int(np.sum(self.weights != 0))) + ')'
            title = title + ' (n=' + str(self.size) + ')'

            if labelpnts:
                for i in range(self.size):
                    ax.text(self.xycoords[:,0][i], self.xycoords[:,1][i], str(int(self.ids[i])), fontsize=10, bbox={'facecolor':'k', 'alpha':0.2, 'pad':1})

            ax.set_title(title)
            # set the limits of the plot to the limits of the data
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')

            ###### plot the frequency distribution
            if raster.density is None:
                raster.computePopulationDistribution()
            raster.computeSampleDistribution([self])
            vals = raster.getData()
            ax2 = fig.add_subplot(gs[1])
            if raster.getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                ## plot population distribution
                den = raster.density[raster.density != 0]
                pos = np.arange(len(np.unique(vals))) + 0.5
                ax2.barh(pos, den, align='center', color = 'gray', label='background')

                ## plot sample distribution (unweighted)
                smp_den = raster.density_sample[0][raster.density != 0]

                ax2.barh(pos, smp_den, align='center', edgecolor = 'red', facecolor = 'red', fill=True, alpha=0.4, label='occurrence')

                if np.std(self.weights) != 0: # if there are unequal weights
                    ## plot sample distribution (weighted)
                    smp_denw = raster.density_sample_weighted[0][raster.density != 0]
                    ax2.barh(pos, smp_denw, align='center', edgecolor = 'y', facecolor = 'y', fill=True, alpha=0.2, label='occurrence_w')

                # the legend
                ax2.legend(loc='best', ncol = 1, frameon=False, fontsize=10)

                labels = []
                for val in np.unique(vals):
                    labels.append(str(int(val)))

                ax2.set_ylim(pos.min() - 0.4, pos.max() + 0.4)
                ax2.set_yticks(pos)
                ax2.set_yticklabels(labels)

                # set the limits of the plot to the limits of the data
                if np.std(self.weights) != 0:
                    max_den = max(den.max(), max(smp_den), max(smp_denw))
                else:
                    max_den = max(den.max(), max(smp_den))
                ax2.set_xlim(0.0, 1.05 * max_den)
                ax2.set_xticks([0, max_den])

            else:
                xmin = raster.min
                xmax = raster.max
                x = np.linspace(xmin, xmax, conf.N_INTERVALS)

                ## plot population distribution
                ax2.plot(raster.density, x, color = 'gray', label='background')

                ## plot sample distribution (unweighted)
                y = raster.density_sample[0]
                ax2.plot(y, x, color = 'red', label='occurrence')

                if np.std(self.weights) != 0:
                    yw = raster.density_sample_weighted[0]
                    ax2.plot(yw, x, color = 'y', label='occurrence_w')

                ax2.legend(loc='best', ncol = 1, frameon=False, fontsize=10)

                # set the limits of the plot to the limits of the data
                if np.std(self.weights) != 0:
                    max_den = max(max(raster.density), max(y), max(yw))
                else:
                    max_den = max(max(raster.density), max(y))
                ax2.set_ylim(vals.min(), vals.max())
                ax2.set_xlim(0, 1.05 * max_den)
                ax2.set_xticks([0, max_den])

                #Plot data points
                sample = util.extractCovariatesAtPoints([raster], self)[0]
                ax2.scatter(0.02 * max_den * np.ones_like(sample), sample, marker = '+', s = 10,
                            color='k', alpha=.5, label=None)

                ax2.scatter(0.02 * max_den * np.ones_like(sample), sample, s = 10*self.weights,
                            color='r', alpha=.5, label=None)

                if labelpnts:
                    for i in range(self.size):
                        ax2.text(0.04 * max_den, sample[i], str(int(self.ids[i])), fontsize=8, color = 'k', alpha=.5)

            ax2.set_xlabel('Density')

            if block: # if block, show plot right now
                plt.show()
            else:     # otherwise, draw plot, show them all together at the end
                plt.draw()
            if conf.SAVEFIG:
                plt.savefig('figs' + os.sep + title + '.png', dpi=300)
                plt.close()

        except Exception as e:
            raise

    def plot_density_curves(self, covariates, covweights = None, flag = None, block = conf.BLOCK_PLOT, labelpnts = conf.LABEL_PNTS, points_bkg=None):
        ''' plot density curves only
        '''
        try:
            __flag = ''
            if flag is not None:
                __flag = flag

            sim1 = 0.0
            sim2 = 0.0

            if covweights is None:
                covweights = 1.0*np.ones(len(covariates))
            covweights = covweights / np.sum(covweights)

            valsAtsamplePoints = util.extractCovariatesAtPoints(covariates, self)

            for i in range(len(covariates)):
                fig = plt.figure(int(time.time()*1000), figsize = (12, 6)) # (8, 6) ) #

                sample = valsAtsamplePoints[i]
                minmax = [covariates[i].min, covariates[i].max]

                #if covariates[i].density_sample is None:
                covariates[i].computeSampleDistribution([self])
                if covariates[i].density is None:
                    covariates[i].computePopulationDistribution(points_bkg=points_bkg)

                if covariates[i].getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:

                    #print(covariates[i].uvals)

                    x = np.arange(minmax[0], minmax[1] + 1, 1)
                    y0 = covariates[i].density
                    #print('1', y0)
                    den = y0[y0 != 0]
                    #print('2', den)
                    pos = np.arange(len(den)) + 0.5
                    plt.bar(pos, den, align='center', color = 'gray', label='background')

                    y1 = covariates[i].density_sample[0]
                    den = y1[y0 != 0]
                    #print('3', den)
                    plt.bar(pos, den, align='center', edgecolor = 'red', facecolor = 'red', fill=True, alpha=0.4, label='occurrence')

                    if np.std(self.weights) != 0:
                        y2 = covariates[i].density_sample_weighted[0]
                        den = y2[y0 != 0]
                        plt.bar(pos, den, align='center', edgecolor = 'y', facecolor = 'y', fill=True, alpha=0.2, label='occurrence_w')
                        maxy = max(max(y0), max(y1), max(y2))
                    else:
                        maxy = max(max(y0), max(y1))

                    labels = []
                    for val in covariates[i].uvals: #np.unique(covariates[i].getData()):
                        labels.append(str(int(val)))

                    plt.xlim(pos.min() - 0.5, pos.max() + 0.5)
                    #print(den)
                    #print(pos)
                    #print(labels)
                    plt.xticks(pos, labels)
                    #sys.exit(0)

                else: # interal/ratio variables
                    #print('hello' , minmax)
                    x = np.linspace(minmax[0], minmax[1], conf.N_INTERVALS)

                    y0 = covariates[i].density
                    plt.plot(x, y0, linewidth = 3, color = 'gray', label = 'background')
                    #print(y0)

                    y1 = covariates[i].density_sample[0]
                    plt.plot(x, y1, linewidth = 3, color = 'r', label = 'occurrence')

                    #print(y1)

                    if np.std(self.weights) != 0:
                        y2 = covariates[i].density_sample_weighted[0]
                        plt.plot(x, y2, linewidth = 3, color = 'y', label = 'occurrence_w')
                        maxy = max(max(y0), max(y1), max(y2))
                    else:
                        maxy = max(max(y0), max(y1))

                    plt.xlim(minmax[0], minmax[1])

                #Plot the samples
                if covariates[i].getMsrLevel() not in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                    #plt.scatter(sample, 0.02 * maxy * np.ones_like(sample), marker = '+', s = 10,
                    #            color='k', alpha=.5, label=None)

                    plt.scatter(sample[self.weights!=0], 0.02 * maxy * np.ones_like(sample[self.weights!=0]), marker = '*', s = 5*self.weights[self.weights!=0],
                                color='red', alpha=.1, label=None)

                    if conf.LABEL_PNTS:
                        for j in range(self.size):
                            plt.text(sample[j], 0.04 * maxy, str(int(self.ids[j])), fontsize=8, color = 'k', alpha=.5)

                else: ## do not plot data points for categorical variables
                    pass

                plt.ylim(0, 1.05 * maxy)
                plt.yticks([0, maxy])
                plt.legend(loc = 'best', ncol = 1, frameon = False)
                plt.xlabel(covariates[i].filename)
                plt.ylabel('density')
                title = 'prob. density distribution on ' + covariates[i].filename
                plt.title(title)

                if block: # if block, show plot right now
                    plt.show()
                else:     # otherwise, draw plot, show them all together at the end
                    plt.draw()

                if conf.SAVEFIG:
                    plt.savefig('figs' + os.sep + title + __flag + '.png', dpi=300)
                    plt.close()

            return sim1, sim2
        except Exception as e:
            raise
