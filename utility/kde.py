# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Oct. 23 2021

import numpy as np
import math, random
import os, time
import matplotlib.pyplot as plt
import conf

import pyopencl as cl
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

########### END OF IMPORT MODULES ##################

class KDE():
    def __init__(self):
        #pass
        self.ctx = None # opencl context
        self.prg = None # opencl program
        self.queue = None # opencl queue
        self.mf = None  # opencl memery flag

        self.data = None
        self.x = None
        self.weights = None
        self.bandwidths = None
        self.density = None
        self.density_inc = None
        self.density_exc = None

        self.data_g = None
        self.x_g = None
        self.weights_g = None
        self.bandwidths_g = None
        self.density_g = None
        self.density_inc_g = None
        self.density_exc_g = None

        self.opencl_already_setup = False

    def __setupOpenCL(self, data = None, weights = None, bandwidths = None,
                    x = None, density = None,
                    density_inc = None, density_exc = None):
        if self.opencl_already_setup:
            return

        ##### config computing platform and device
        DEVICE = None
        FOUND_FIRST = False
        for platform in cl.get_platforms():
            if platform.name == conf.OPENCL_CONFIG['Platform']:
                PLATFORM = platform
                # Print each device per-platform
                for device in platform.get_devices():
                    if device.name == conf.OPENCL_CONFIG['Device']:
                        DEVICE = device
                        FOUND_FIRST = True
                        #print("***1***")
                        break
            if FOUND_FIRST:
                break
        print(DEVICE.name, PLATFORM.name)

        # opencl context
        self.ctx = cl.Context([DEVICE])
        #print("***2***")
        # opencl command queue
        self.queue = cl.CommandQueue(self.ctx)
        #print("***3***")
        ##### build opencl kernel from code in the file
        f = open(conf.KDE_KERNEL_FN, 'r')
        fstr = "".join(f.readlines())
        self.prg = cl.Program(self.ctx, fstr).build()

        ##### allocate memory space on device
        self.mf = cl.mem_flags

        if data is not None:# and self.data is None:
            self.data = np.copy(data).astype(np.float32)
            self.data_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.data)

        if weights is not None:
            self.weights = np.copy(weights).astype(np.float32)
            self.weights_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.weights)

        if bandwidths is not None:
            self.bandwidths = np.copy(bandwidths).astype(np.float32)
            self.bandwidths_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.bandwidths)

        if x is not None:# and self.x is None:
            self.x = np.copy(x).astype(np.float32)
            self.x_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.x)

        if density is not None and self.density is None:
            self.density = np.copy(density).astype(np.float32)
            self.density_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.density.nbytes)

        if density_inc is not None and self.density_inc is None:
            self.density_inc = np.copy(density_inc).astype(np.float32)
            self.density_inc_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.density_inc.nbytes)

        if density_exc is not None and self.density_exc is None:
            self.density_exc = np.copy(density_exc).astype(np.float32)
            self.density_exc_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.density_exc.nbytes)

        self.opencl_already_setup = True
        #print 'Done setting up OpenCL...'

    def updateOpenCLData(self, data = None, weights = None, bandwidths = None,
                    x = None, density = None,
                    density_inc = None, density_exc = None):

        if not self.opencl_already_setup:
            #print 'Need to call setupOpenCL() first...'
            self.__setupOpenCL(data = data, weights = weights, bandwidths = bandwidths,
                            x = x, density = density,
                            density_inc = density_inc, density_exc = density_exc)
            return

        if data is not None:# and self.data is None:
            self.data = np.copy(data).astype(np.float32)
            self.data_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.data)

        if weights is not None:
            self.weights = np.copy(weights).astype(np.float32)
            self.weights_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.weights)

        if bandwidths is not None:
            self.bandwidths = np.copy(bandwidths).astype(np.float32)
            self.bandwidths_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.bandwidths)

        if x is not None:# and self.x is None:
            self.x = np.copy(x).astype(np.float32)
            self.x_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.x)

        if density is not None and self.density is None:
            self.density = np.copy(density).astype(np.float32)
            self.density_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.density.nbytes)

        if density_inc is not None and self.density_inc is None:
            self.density_inc = np.copy(density_inc).astype(np.float32)
            self.density_inc_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.density_inc.nbytes)

        if density_exc is not None and self.density_exc is None:
            self.density_exc = np.copy(density_exc).astype(np.float32)
            self.density_exc_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.density_exc.nbytes)

        #print 'Done updating OpenCL data...'

    # weighted average
    def __weightedMean(self, values, weights = None):
        ''' return weighted average
            values: sample
            weights: sample weights
        '''
        #print 'vals:', len(values)
        #if weights is not None: print 'ws:', len(weights)
        return np.average(values, weights=weights)

    # weighted std
    def __weightedStd(self, values, weights = None):
        ''' return weighted std
            values: sample
            weights: sample weights
        '''
        #print 'weights', weights
        values = np.array(values)
        average = self.__weightedMean(values, weights=weights)
        #print 'average', average
        variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
        #print 'variance', variance
        return math.sqrt(variance)

    # rule-of-thumb bandwidth (weighted)
    # sample points in data
    def __bandwidth(self, data, weights=None, neff_only = False):
        if weights is None:
            weights = np.ones_like(data)

        neff = np.sum(weights)**2/np.sum(weights**2)
        #print("neff = %f" % neff)
        #print("neff_only = %f" % np.power(neff*0.75, -0.2))
        tmp = 1.06 * self.__weightedStd(data, weights = weights) * (neff) ** (-0.2)
        #print(type(tmp))
        #print("!neff_only = %f" % tmp)
        if neff_only:
            return np.power(neff*0.75, -0.2)
        else:
            return 1.06 * self.__weightedStd(data, weights = weights) * (neff) ** (-0.2)

    # kernel density estimation (adaptive bandwidths) (weighted)
    # at data points in x given sample points in data
    def adaptive_kde(self, data, x, weights = None, opencl = conf.OPENCL_KDE):
        if weights is None:
            weights = np.ones_like(data)
        if opencl:
            weights = weights.astype(np.float32)

            density_inc = np.zeros_like(data).astype(np.float32)
            density_exc = np.zeros_like(data).astype(np.float32)

            self.updateOpenCLData(data = data, weights = weights, density_inc = density_inc, density_exc = density_exc)

            M = np.int32(len(data))
            #h = self.__bandwidth(data, weights)
            h = self.__golden_section_search(data, weights, opencl=True)
            h = np.float32(h)
            #print 'h_cl =', h

            self.prg.kde_densityAtDataPoints(self.queue, self.data.shape, None, M, h, self.data_g, self.weights_g, self.density_inc_g, self.density_exc_g)
            ## copy result data
            cl.enqueue_copy(self.queue, self.density_inc, self.density_inc_g)
            #cl.enqueue_copy(self.queue, self.density_exc, self.density_exc_g)

            g = np.sum(np.log(self.density_inc))
            #g = np.exp(g / np.sum(weights))
            g = np.exp(g / M)
            #print 'g =', g

            bandwidths = h * np.power(self.density_inc/g, -0.5).astype(np.float32)
            density = np.zeros_like(x).astype(np.float32)
            self.updateOpenCLData(x = x, bandwidths = bandwidths, density = density)

            self.prg.kde_adaptiveKDE(self.queue, self.x.shape, None, M, self.data_g, self.bandwidths_g, self.weights_g, self.x_g, self.density_g)
            ## copy result data
            cl.enqueue_copy(self.queue, self.density, self.density_g)

            return np.copy(self.density)

        else:
            print('### native Python adaptive_kde()')
            #h = self.__bandwidth(data, weights = weights) # global bandwidth
            h = self.__golden_section_search(data, weights, opencl = False) # global bandwidth
            #print 'h_cpu =', h
            # pilot density estimation, needed to compute adaptive bandwidths
            denPnts = [] # density at data points
            g = 0.0
            for j in range(len(data)):
                xi = data[j]
                s = 0.0
                nn = 0.0
                for i in range(len(data)):
                    di = data[i]
                    wi = 1
                    if weights is not None: wi = weights[i]
                    nn += wi
                    tmp = math.exp(-(xi - di)**2/(2*h**2))/(math.sqrt(2*math.pi))
                    s += tmp * wi / h
                denPnts.append(s/nn)
                g += math.log(s/nn)

            # geometric mean
            #g = math.exp(g / nn)
            g = math.exp(g / len(data))
            #print 'g =', g
            # adaptive bandwidths i.e. hi = h * pow(denPnts[i]/g, -0.5)
            den = []
            for xi in x:
                s = 0.0
                nn = 0
                for i in range(len(data)):
                    di = data[i]
                    wi = weights[i]
                    nn += wi
                    hi = h * pow(denPnts[i]/g, -0.5)
                    tmp = math.exp(-(xi - di)**2/(2*hi**2))/(math.sqrt(2*math.pi))
                    s += tmp * wi / hi
                den.append(s/nn)
            return np.array(den)

    # kernel density estimation (fixed rule-of-thumb bandwidth) (weighted)
    # at data points in x given sample points in data
    # flips is a 0/1 bits array indicating whether a sample point is included
    def simple_kde(self, data, x, weights = None, opencl = conf.OPENCL_KDE):
        h0 = self.__bandwidth(data, weights)#, neff_only = False)
        #print(weights)
        #print("simple_kde h0: %f" % h0)
        #print(data)
        if weights is None:
            weights = np.ones_like(data)
        if opencl: ### compute on CPU/GPU using opencl
            weights = weights.astype(np.float32)
            density = np.zeros_like(x).astype(np.float32)

            #t0 = time.time()
            self.updateOpenCLData(data = data, weights = weights, x = x, density = density)
            #print 'simple_kde - host to device transfer took', (time.time() - t0) * 1000, 'ms'

            M = np.int32(len(data))
            h = np.float32(h0)
            #print 'h_cl =', h


            #print 'self.data', len(self.data), self.data
            #print 'self.x', len(self.x), self.x
            #print 'self.density', len(self.density), self.density
            #t0 = time.time()
            self.prg.kde_fixedKDE(self.queue, self.x.shape, None, M, h, self.data_g, self.weights_g, self.x_g, self.density_g)
            ## copy result data
            cl.enqueue_copy(self.queue, self.density, self.density_g)
            #print 'simple_kde - compute and transfer back took', (time.time() - t0) * 1000, 'ms'
            #print 'self.density_g', len(self.density_g), self.density_g

            #print(self.density)
            return np.copy(self.density)

        else:
            #'''
            # Native Python implementation, inefficient
            #h0 = self.__bandwidth(data, weights = weights) # rule-of-thumb bandwidth
            h = h0
            print('### native Python simple_kde()')
            den = [] # density at data points
            for xi in x:
                s = 0.0
                nn = 0
                for i in range(len(data)):
                    di = data[i]
                    wi = 1.0
                    if weights is not None: wi = weights[i]
                    nn += wi
                    tmp = math.exp(-(xi - di)**2/(2*h**2))/(math.sqrt(2*math.pi))
                    s += tmp * wi / h
                den.append(s/nn)
            return np.array(den)
            #'''
    def __log_likelihood(self, data, h, weights = None, opencl = conf.OPENCL_KDE):

        if weights is None:
            weights = np.ones_like(data)

        if opencl:
            weights = weights.astype(np.float32)

            density_inc = np.zeros_like(data).astype(np.float32)
            density_exc = np.zeros_like(data).astype(np.float32)

            self.updateOpenCLData(data = data, weights = weights, density_inc = density_inc, density_exc = density_exc)

            M = np.int32(len(data))
            h = np.float32(h)

            self.prg.kde_densityAtDataPoints(self.queue, self.data.shape, None, M, h, self.data_g, self.weights_g, self.density_inc_g, self.density_exc_g)
            ## copy result data
            cl.enqueue_copy(self.queue, self.density_exc, self.density_exc_g)
            #print 'cl ** h =', h, 'den_exc =', self.density_exc
            l = 0.0
            for j in range(len(self.density_exc)):
                l += math.log(max(self.density_exc[j], 1e-8))
            #l = np.sum(np.log(np.min(np.array([self.density_exc, 1e-8 * np.ones_like(self.density_exc)]), axis = 0)))
            #print 'cl ** h =', h, 'L =', l
            return np.copy(self.density_exc), l

        else:
            print('### native Python __log_likelihood()')
            L = 0.0
            den = []
            n = len(data)
            if weights is not None:
                n = np.sum(weights)
            for j in range(len(data)):
                xj = data[j]
                s = 0.0
                for i in range(len(data)):
                    if i != j:
                        di = data[i]
                        wi = weights[i]
                        tmp = math.exp(-(xj - di)**2/(2*h**2))/(math.sqrt(2*math.pi))
                        s = s + wi * tmp / h
                den.append(s/n)
                L = L + math.log(max(s/(n-weights[j]), 1e-17))

            return np.array(den), L

    def cv_kde(self, data, x, weights = None, opencl = conf.OPENCL_KDE):
        h = self.__golden_section_search(data, weights = weights, opencl = opencl) # global bandwidth
        #print 'h_cv =', h
        if weights is None:
            weights = np.ones_like(data)
        if opencl:
            weights = weights.astype(np.float32)
            density = np.zeros_like(x).astype(np.float32)

            self.updateOpenCLData(data = data, weights = weights, x = x, density = density)

            M = np.int32(len(data))
            h = np.float32(h)

            self.prg.kde_fixedKDE(self.queue, self.x.shape, None, M, h, self.data_g, self.weights_g, self.x_g, self.density_g)
            ## copy result data
            cl.enqueue_copy(self.queue, self.density, self.density_g)

            return np.copy(self.density)

        else:
            print('### native Python cv_kde()')
            den = [] # density at data points
            for xi in x:
                s = 0.0
                nn = 0
                for i in range(len(data)):
                    di = data[i]
                    wi = weights[i]
                    nn += wi
                    tmp = math.exp(-(xi - di)**2/(2*h**2))/(math.sqrt(2*math.pi))
                    s += tmp * wi / h
                den.append(s/nn)

            return np.array(den)

    def __golden_section_search(self, data, weights = None, opencl = conf.OPENCL_KDE):
        h0 = self.__bandwidth(data, weights)
        hA = h0 / 10
        hD = 2 * h0
        width = hD - hA
        eps = width / 50
        factor = 1 + math.sqrt(5.0)
        print('h0=%.3f, hA=%.3f, hD=%.3f, eps=%.3f' % (h0, hA,hD, eps))
        counter = 0
        #print '----------'
        #print '(', hA, hD, ')'
        while width > eps:
            hB = hA + width/factor
            hC = hD - width/factor

            den, LB = self.__log_likelihood(data, hB, weights, opencl)
            den, LC = self.__log_likelihood(data, hC, weights, opencl)

            if LB > LC:
                hD = hC
                #print hB, LB
            else:
                hA = hB
                #print hC, LC

            width = hD - hA
            #print counter
            counter += 1
            print('iteration %d, hA=%.3f, hD=%.3f' % (counter, hA, hD))
        return 0.5 * (hA + hD)

    def evaluate(self, data, x, weights = None, opencl = conf.OPENCL_KDE, bwoption=0):
        '''
        '''
        print('...in kde: %s %d %s' % (opencl, bwoption, conf.OPENCL_CONFIG))
        if bwoption == 0:
            return self.simple_kde(data, x, weights = weights, opencl = opencl) ### only for habitat suitability mapping
        elif bwoption == 1:
            return self.cv_kde(data, x, weights = weights, opencl = opencl)
        else:
            return self.adaptive_kde(data, x, weights = weights, opencl = opencl)
