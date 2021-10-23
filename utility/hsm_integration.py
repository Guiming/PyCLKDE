# Author: Guiming Zhang
# Last update: Oct. 23 2021

import os, time, sys, json, platform
import numpy as np
from scipy import stats
import matplotlib
#print(matplotlib.get_backend())
#matplotlib.use("Qt4Agg")
#print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import raster, points, util, evalmetric, conf

class HSM:
    ''' a class implementing KDE based wildlife habitat suitability mapping (HSM);
            Zhang, G., Zhu, A.-X., Windels, S.K., Qin, C.-Z., 2018.
            Modelling species habitat suitability from presence-only data using kernel density estimation.
            Ecol. Indic. 93, 387 396.
        it implements data-, knowledge-, and model-level data integration for HSM;
            Zhang, G., Zhu, A., He, Y., Huang, Z., Ren, G., Xiao, W., 2020.
            Integrating multi-source data for wildlife habitat mapping: A case study of the black-and-white snub-nosed monkey (Rhinopithecus bieti) in Yunnan, China.
            Ecol. Indic. 118, 106735.
    '''
    __sigTimestampe = None
    __covariates = None # a list of covariates raster
    __covweights = None # weights for each covariate
    __samples = None    # sample points
    __aggstrategy = 'weighted average' # strategy to integrate suitability on predictors
    __samplepdfonly = False  # compute suitability solely based on sample pdf
                                       # either 'weighted average' (default) or 'limiting factor'
    __integration_level = 'data' # 'data': data level integration
                                 # 'knowledge': knowledge level integration
                                 # 'model': model level integration
    __integration_strategy = 'mean' # could also be minimum, or maximum

    __suitabilitymap = None # the predicted suitability map
    __suitsAtsamplePoints = None # suitability values at sample points

    __valsAtbkgPoints = None # covariates values at bkg points

    def __init__(self, covariates, samples, covweights=None, samplepdfonly=False, aggstrategy = 'weighted average', \
                        integration_level = 'data', integration_strategy = 'mean', points_bkg = None, sample_pdf_update=False, sample_index=0):
        self.__sigTimestampe = int(time.time()*1000) # in ms
        self.__sample_index = sample_index ## this is used to avoid repeatedly computing sample pdf (knowledge/model-level integration)
        self.__covariates = covariates

        self.__samples = []
        for sample_set in samples: self.__samples.append(sample_set.copySelf())

        self.__covweights = covweights
        self.__aggstrategy = aggstrategy
        self.__samplepdfonly = samplepdfonly

        self.__integration_level = integration_level
        self.__integration_strategy = integration_strategy

        if covweights is None:
            self.__covweights = np.ones(len(self.__covariates))

        ### compute sample distributions (weighted samples as well as unweighted samples)
        #stats = []

        #for covariate in self.__covariates:
        #    covariate.resetTimes()
        #util.resetTimes()

        for covariate in self.__covariates:
            if covariate.density is None:
                covariate.computePopulationDistribution(points_bkg=points_bkg)

        if len(self.__samples) == 1:
            for covariate in self.__covariates:
                if covariate.density_sample is None or sample_pdf_update:
                    covariate.computeSampleDistribution(self.__samples)

                min = covariate.min
                max = covariate.max

                if covariate.msrInt == 0 or covariate.msrInt == 1:
                    x = range(int(min), int(max) + 1)
                else:
                    x = np.arange(min, max, (max-min)/conf.N_INTERVALS)

                #print np.array(x).shape, np.array(covariate.density_sample).shape, np.array(covariate.density).shape
                #if 'BIO1.tif' in covariate.filename:
                #    np.savetxt('data' + os.sep + covariate.filename + '.csv', np.vstack((np.array(x), np.array(covariate.density), np.array(covariate.density_sample))).T, delimiter=',')

        if len(self.__samples) > 1 and self.__integration_level == 'data': # need to integrating two sources of data
            int_samples = self.__samples[0]
            for i in range(1, len(self.__samples)):
                int_samples = int_samples.mergePoints(self.__samples[i]).eliminateDuplicates()
            for covariate in self.__covariates:
                if covariate.density_sample is None or sample_pdf_update:
                    covariate.computeSampleDistribution([int_samples])


                min = covariate.min
                max = covariate.max
                if covariate.msrInt == 0 or covariate.msrInt == 1:
                    x = range(int(min), int(max) + 1)
                else:
                    x = np.arange(min, max, (max-min)/conf.N_INTERVALS)

                #print np.array(x).shape, np.array(covariate.density_sample).shape, np.array(covariate.density).shape
                #np.savetxt('data' + os.sep + covariate.filename + '.csv', np.vstack((np.array(x), np.array(covariate.density), np.array(covariate.density_sample))).T, delimiter=',')

        if len(self.__samples) > 1 and self.__integration_level in ['knowledge', 'model']: # == 'knowledge': #
                for covariate in self.__covariates:
                    if covariate.density_sample is None or sample_pdf_update:
                        covariate.computeSampleDistribution(self.__samples)

                    #'''
                    min = covariate.min
                    max = covariate.max
                    if covariate.msrInt == 0 or covariate.msrInt == 1:
                        x = range(int(min), int(max) + 1)
                    else:
                        x = np.arange(min, max, (max-min)/conf.N_INTERVALS)
                    #'''
                    #print np.array(x).shape, np.array(covariate.density_sample).shape, np.array(covariate.density).shape
                    #if 'BIO1.tif' in covariate.filename:
                    #    np.savetxt('data' + os.sep + covariate.filename + '_bw' + str(conf.BW_OPTION_OCCR) + '.csv', np.vstack((np.array(x), np.array(covariate.density), np.array(covariate.density_sample))).T, delimiter=',')

        #for covariate in self.__covariates:
        #    print(covariate.time_acc_reads, covariate.time_acc_writes, covariate.time_acc_kde_bkg, covariate.time_acc_kde_smpl)

    def __del__(self):
        '''
        '''
        pass
    def __suitabilityMap(self, samplepdfonly=None, covweights=None, aggstrategy=None):
        ''' generate a habitat suitability map for the whole study area
            given the weights on each predictor and the aggregation strategy
        '''
        if samplepdfonly is not None:
            self.__samplepdfonly = samplepdfonly
        if covweights is not None:
            self.__covweights = covweights
        if aggstrategy is not None:
            self.__aggstrategy = aggstrategy

        suitabilities = np.zeros_like(self.__covariates[0].getData()) # suitabilities based on unweighted samples

        for i in range(len(self.__covweights)):
            covariate = self.__covariates[i]
            w = self.__covweights[i]/np.sum(self.__covweights)
            pdf_poplation = covariate.density
            pdf_sample = covariate.density_sample
            pdf_sample_w = covariate.density_sample_weighted

            vals = covariate.getData()
            xmin = covariate.min
            xmax = covariate.max

            if covariate.getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                idx = (vals - xmin).astype(int)
            else:
                idx = ((vals - xmin)/(xmax - xmin) * (conf.N_INTERVALS - 1)).astype(int)

            if (len(pdf_sample) == 1) or (len(pdf_sample) > 1 and self.__integration_level == 'data'):
                pdf_p = pdf_poplation[idx]
                #pdf_s = pdf_sample[0][idx]
                pdf_s = pdf_sample[self.__sample_index][idx]
                suit = 1.0 * np.zeros_like(pdf_p)
                if self.__samplepdfonly:
                    ## sample distribution only
                    suit = pdf_s / pdf_s.max()
                else:
                    ## consider background distribution
                    suit[pdf_p != 0.0] = pdf_s[pdf_p != 0.0] / pdf_p[pdf_p != 0.0]
                    suit = 1.0 / (1.0 + np.exp(-1.0 * (suit - 1.0)))

            if len(pdf_sample) > 1 and self.__integration_level == 'knowledge':
                pdf_p = pdf_poplation[idx]
                pdf_sample_idx = []
                for pdf in pdf_sample:
                    pdf_sample_idx.append(pdf[idx])

                suit = self.integration(pdf_sample_idx, pdf_p, self.__samplepdfonly, self.__integration_strategy)

            if self.__aggstrategy == 'weighted average':
                suitabilities = suitabilities + w * suit
            else: # 'limiting factor'
                w = 1.0
                if i == 0:
                    suitabilities = w * suit
                else:
                    suitabilities = np.min([suitabilities, w * suit], axis = 0)

        suit_raster = self.__covariates[0].copySelf()
        suit_raster.filename = 'suitability'
        suit_raster.setMsrLevel(conf.MSR_LEVEL_RATIO)
        suit_raster.updateRasterData(suitabilities)

        self.__suitabilitymap = [suit_raster]
        return self.__suitabilitymap


    def suitabilityMap(self, samplepdfonly=None, covweights=None, aggstrategy=None):
        if self.__integration_level in ['data', 'knowledge']: ## data or knowledge level integration
            return self.__suitabilityMap(samplepdfonly, covweights, aggstrategy)
        else: ## model level integration
            final_suitmap = None
            cnt = 0
            #for samples, i in zip(self.__samples, list(range(0, len(self.__samples)))):
            for i in list(range(0, len(self.__samples))):
                samples = self.__samples[i]
                tmp_hsm = HSM(self.__covariates, [samples], self.__covweights, self.__samplepdfonly, self.__aggstrategy,\
                                    'data', self.__integration_strategy, sample_index=i)
                suitmap = tmp_hsm.__suitabilityMap(samplepdfonly, covweights, aggstrategy)[0]

                #suitmap.plot(title = 'hsm' + str(cnt))
                #suitmap.writeAscii('hsm' + str(cnt) + '.asc')

                if final_suitmap is None:
                    final_suitmap = suitmap
                else:
                    final_suitmap = final_suitmap.integrate(suitmap, self.__integration_strategy)
                cnt += 1
        #final_suitmap.plot(title = 'hsm')
        #final_suitmap.writeAscii('hsm.asc')
        return [final_suitmap]

    def __suitabilityMapTile(self, samplepdfonly=None, covweights=None, aggstrategy=None, suitabilityfn='suitabilitytile.tif'):
        ''' generate a habitat suitability map for the whole study area
            given the weights on each predictor and the aggregation strategy
        '''
        if samplepdfonly is not None:
            self.__samplepdfonly = samplepdfonly
        if covweights is not None:
            self.__covweights = covweights
        if aggstrategy is not None:
            self.__aggstrategy = aggstrategy

        for i in range(len(self.__covariates)):
            #self.__covariates[i].tileReader.reset()
            self.__covariates[i].resetTileReader()

        suit_raster = self.__covariates[0].copySelf()
        suit_raster.filenamefull = suitabilityfn
        suit_raster.filename = os.path.basename(suitabilityfn)
        suit_raster.setMsrLevel(conf.MSR_LEVEL_RATIO)

        template_raster = raster.Raster() # a raster made up with only the current tile
        suit_raster.readRasterGDAL()
        data = suit_raster.getData2D()
        while data is not None:
            suitabilities2D = 1.0 * np.copy(data)
            template_raster.createRaster(suitabilities2D, xoff=suit_raster.tile_xoff, yoff=suit_raster.tile_yoff, \
                                                geotransform = suit_raster.geotransform, \
                                                projection = suit_raster.tileReader.projection, \
                                                nodata = suit_raster.nodatavalue)

            if suit_raster.getData().size > 0:
                #print("suit_raster.getData().size", suit_raster.getData().size)
                suitabilities = np.zeros_like(suit_raster.getData()) # suitabilities based on unweighted samples
                #print("suitabilities.shape: ", suitabilities.shape)
                for i in range(len(self.__covweights)):
                    covariate = self.__covariates[i]
                    w = self.__covweights[i]/np.sum(self.__covweights)
                    pdf_poplation = covariate.density
                    pdf_sample = covariate.density_sample
                    pdf_sample_w = covariate.density_sample_weighted
                    xmin = covariate.min
                    xmax = covariate.max

                    covariate.readRasterGDAL()

                    if conf.MASK_ON_FLY:
                        tmpdata = covariate.getData2D()
                        #print("tmpdata: ", np.sum(tmpdata != covariate.nodatavalue))
                        tmpdata[data == suit_raster.nodatavalue] = covariate.nodatavalue
                        #print("**tmpdata: ", np.sum(tmpdata != covariate.nodatavalue))
                        covariate.updateRasterData2D(tmpdata)

                    vals = covariate.getData()
                    #print("**covariate.getData().shape: ", covariate.getData().shape)

                    #print("vals:", vals)
                    #print("xmin:", xmin)
                    #print("xmax:", xmax)

                    if covariate.getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                        idx = (vals - xmin).astype(int)
                    else:
                        idx = ((vals - xmin)/(xmax - xmin) * (conf.N_INTERVALS - 1)).astype(int)

                    if (len(pdf_sample) == 1) or (len(pdf_sample) > 1 and self.__integration_level == 'data'):
                        #print("idx:", idx)
                        pdf_p = pdf_poplation[idx]
                        #print("pdf_p:", pdf_p)
                        #pdf_s = pdf_sample[0][idx]
                        pdf_s = pdf_sample[self.__sample_index][idx]
                        #print("pdf_sample[0]:", pdf_sample[0])
                        #print("pdf_s:", pdf_s)
                        suit = 1.0 * np.zeros_like(pdf_p)
                        if self.__samplepdfonly:
                            ## sample distribution only
                            suit = pdf_s / pdf_s.max()
                        else:
                            ## consider background distribution
                            suit[pdf_p != 0.0] = pdf_s[pdf_p != 0.0] / pdf_p[pdf_p != 0.0]
                            suit = 1.0 / (1.0 + np.exp(-1.0 * (suit - 1.0)))

                    if len(pdf_sample) > 1 and self.__integration_level == 'knowledge':
                        pdf_p = pdf_poplation[idx]
                        pdf_sample_idx = []
                        for pdf in pdf_sample:
                            pdf_sample_idx.append(pdf[idx])

                        suit = self.integration(pdf_sample_idx, pdf_p, self.__samplepdfonly, self.__integration_strategy)

                    if self.__aggstrategy == 'weighted average':
                        suitabilities = suitabilities + w * suit
                    else: # 'limiting factor'
                        w = 1.0
                        if i == 0:
                            suitabilities = w * suit
                        else:
                            suitabilities = np.min([suitabilities, w * suit], axis = 0)
                # now need to reform suitabilities to a 2D array
                template_raster.updateRasterData(suitabilities)
            else:
                for i in range(len(self.__covariates)):
                    self.__covariates[i].readRasterGDAL()
                #pass
            suit_raster.writeRasterGDAL(suitabilityfn, tile_data = template_raster.getData2D(), tile_xoff = suit_raster.tile_xoff, tile_yoff = suit_raster.tile_yoff)

            suit_raster.readRasterGDAL()
            data = suit_raster.getData2D()

    def suitabilityMapTile(self, samplepdfonly=None, covweights=None, aggstrategy=None, suitabilityfn='suitabilitytile.tif'):
        if self.__integration_level in ['data', 'knowledge']: ## data or knowledge level integration
            return self.__suitabilityMapTile(samplepdfonly, covweights, aggstrategy, suitabilityfn)
        else: ## model level integration

            for i in range(len(self.__covariates)):
                #self.__covariates[i].tileReader.reset()
                self.__covariates[i].resetTileReader()

            suit_raster = self.__covariates[0].copySelf()
            suit_raster.filenamefull = suitabilityfn
            suit_raster.filename = os.path.basename(suitabilityfn)
            suit_raster.setMsrLevel(conf.MSR_LEVEL_RATIO)

            #template_raster = raster.Raster() # a raster made up with only the current tile
            suit_raster.resetTileReader()
            suit_raster.readRasterGDAL()
            data = suit_raster.getData2D()

            while data is not None:

                #print("data.shape: ", data.shape, suit_raster.tile_xoff, suit_raster.tile_yoff, suit_raster.tile_xsize, suit_raster.tile_ysize)

                suitabilities2D = 1.0 * np.copy(data)
                #template_raster.createRaster(suitabilities2D, xoff=suit_raster.tile_xoff, yoff=suit_raster.tile_yoff, \
                #                                    geotransform = suit_raster.geotransform, \
                #                                    projection = suit_raster.tileReader.projection, \
                #                                    nodata = suit_raster.nodatavalue)
                if suit_raster.getData().size > 0:

                    ## create temporary raster containing only the current tile
                    covariates_tile = []
                    for i in range(len(self.__covariates)):
                        covariate = self.__covariates[i]

                        covariate.readRasterGDAL()
                        data_tile = covariate.getData2D()

                        #print("data_tile.shape: ", data_tile.shape, covariate.tile_xoff, covariate.tile_yoff, covariate.tile_xsize, covariate.tile_ysize)
                        if conf.MASK_ON_FLY:
                            data_tile[data==suit_raster.nodatavalue] = covariate.nodatavalue

                        covariate_tile = raster.Raster(msrlevel=covariate.getMsrLevel())

                        ## should keep the statistics and pdfs of the original raster to ensure correctness
                        covariate_tile.createRaster(data_tile, xoff=covariate.tile_xoff, yoff=covariate.tile_yoff, \
                                                            geotransform = covariate.geotransform, \
                                                            projection = covariate.tileReader.projection, \
                                                            nodata = covariate.nodatavalue, \
                                                            statistics = [covariate.min, covariate.max, covariate.mean, covariate.std], \
                                                            density = covariate.density, \
                                                            density_sample = covariate.density_sample)
                        covariates_tile.append(covariate_tile)

                    ################3
                    final_suitmap = None
                    #for samples in self.__samples:
                    for i in list(range(len(self.__samples))):
                        samples = self.__samples[i]
                        ## sample_pdf_update=False so pdfs do not get recomputed using only the tile and sample points within the tile (which would mess things up)
                        tmp_hsm = HSM(covariates_tile, [samples], self.__covweights, self.__samplepdfonly, self.__aggstrategy,\
                                            'data', self.__integration_strategy, sample_index=i)
                        suitmap = tmp_hsm.__suitabilityMap(samplepdfonly, covweights, aggstrategy)[0]

                        #suitmap.plot(title = 'hsm' + str(cnt))
                        #suitmap.writeAscii('hsm' + str(cnt) + '.asc')

                        #print(suitmap.getData())

                        if final_suitmap is None:
                            final_suitmap = suitmap

                        else:
                            final_suitmap = final_suitmap.integrate(suitmap, self.__integration_strategy)
                    ##########
                    # now need to reform suitabilities to a 2D array
                    suitabilities2D = final_suitmap.getData2D()
                else:
                    for i in range(len(self.__covariates)):
                        self.__covariates[i].readRasterGDAL()

                suit_raster.writeRasterGDAL(suitabilityfn, tile_data = suitabilities2D, tile_xoff = suit_raster.tile_xoff, tile_yoff = suit_raster.tile_yoff)

                suit_raster.readRasterGDAL()
                data = suit_raster.getData2D()

    def __suitabilityAtPoints(self, points = None, bkg_flag = False, samplepdfonly=None, covweights=None, aggstrategy=None):
        ''' compute suitabilities only at points
            if points is None, compute suitabilities at sample points
        '''
        if points is None:
            points = self.__samples[0]
            bkg_flag = False

        if self.__valsAtbkgPoints is None and bkg_flag: # this is tricky
            self.__valsAtbkgPoints = util.extractCovariatesAtPoints(self.__covariates, points)
            valsAtPoints = self.__valsAtbkgPoints
            #print 'exaction ONCE'

        elif self.__valsAtbkgPoints is not None and bkg_flag:
            valsAtPoints = self.__valsAtbkgPoints
            #print 'reuse ONCE'

        else:
            valsAtPoints = util.extractCovariatesAtPoints(self.__covariates, points)
            #print 'NOT reused ONCE'

            #print valsAtPoints

        if samplepdfonly is not None:
            self.__samplepdfonly = samplepdfonly
        if covweights is not None:
            self.__covweights = covweights
        if aggstrategy is not None:
            self.__aggstrategy = aggstrategy

        suitabilities = np.zeros_like(points.attributes) # suitabilities based on unweighted samples

        for i in range(len(self.__covweights)):
            covariate = self.__covariates[i]
            w = self.__covweights[i]/np.sum(self.__covweights)
            pdf_poplation = covariate.density
            pdf_sample = covariate.density_sample
            pdf_sample_w = covariate.density_sample_weighted

            vals = valsAtPoints[i]

            #print(len(vals), np.sum(vals==-9999))

            xmin = covariate.min
            xmax = covariate.max

            if covariate.getMsrLevel() in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
                idx = (vals - xmin).astype(int)
            else:
                idx = ((vals - xmin)/(xmax - xmin) * (conf.N_INTERVALS - 1)).astype(int)

            #print(covariate.filename, xmin, xmax)
            #print(vals[0:10])
            #print(idx[0:10])

            if (len(pdf_sample) == 1) or (len(pdf_sample) > 1 and self.__integration_level == 'data'):
                pdf_p = pdf_poplation[idx]
                pdf_s = pdf_sample[self.__sample_index][idx]
                suit = 1.0 * np.zeros_like(pdf_p)
                if self.__samplepdfonly:
                    ## sample distribution only
                    suit = pdf_s / pdf_s.max()
                else:
                    ## consider background distribution
                    suit[pdf_p != 0.0] = pdf_s[pdf_p != 0.0] / pdf_p[pdf_p != 0.0]
                    suit = 1.0 / (1.0 + np.exp(-1.0 * (suit - 1.0)))

            if len(pdf_sample) > 1 and self.__integration_level == 'knowledge':
                pdf_p = pdf_poplation[idx]
                pdf_sample_idx = []
                for pdf in pdf_sample:
                    pdf_sample_idx.append(pdf[idx])

                suit = self.integration(pdf_sample_idx, pdf_p, self.__samplepdfonly, self.__integration_strategy)

            #np.savetxt('data' + os.sep + covariate.filename + '_suit.csv', np.vstack((np.array(vals), suit)).T, delimiter=',')

            if self.__aggstrategy == 'weighted average':
                suitabilities = suitabilities + w * suit
            else: # 'limiting factor'
                w = 1.0
                if i == 0:
                    suitabilities = w * suit
                else:
                    suitabilities = np.min([suitabilities, w * suit], axis = 0)
                #print w * suit

        if points is None:
            self.__suitsAtsamplePoints = [suitabilities]

        #print suitabilities
        return [suitabilities]

    def suitabilityAtPoints(self, points = None, bkg_flag = False, samplepdfonly=None, covweights=None, aggstrategy=None):
        if self.__integration_level in ['data', 'knowledge']: ## data or knowledge level integration
            return self.__suitabilityAtPoints(points, bkg_flag, samplepdfonly, covweights, aggstrategy)
        else: ## model level integration
            final_suit = None
            #for samples in self.__samples:
            for i in range(0, len(self.__samples)):
                samples = self.__samples[i]
                tmp_hsm = HSM(self.__covariates, [samples], self.__covweights, self.__samplepdfonly, self.__aggstrategy,\
                                    'data', self.__integration_strategy, sample_index=i)
                suit = tmp_hsm.__suitabilityAtPoints(points, bkg_flag, samplepdfonly, covweights, aggstrategy)[0]

                if final_suit is None:
                    final_suit = suit
                else:
                    if self.__integration_strategy == 'minimum':
                        final_suit = np.min([suit, final_suit], axis = 0)
                    if self.__integration_strategy == 'mean':
                        final_suit = np.mean([suit, final_suit], axis = 0)
                    if self.__integration_strategy == 'maximum':
                        final_suit = np.max([suit, final_suit], axis = 0)
        return [final_suit]

    def plot_roc_curve(self, test_samples = None, bkg_samples = None, number_of_background_samples = 1000, title = 'roc curve', covweights=None, aggstrategy=None, samplepdfonly = None):
        ''' plot roc curve, compute auc
            if test_samples is None, plot training roc
        '''
        if bkg_samples is None:
            bkg_samples = points.Points()
            t0 = time.time()
            bkg_samples.generateRandom(number_of_background_samples, self.__suitabilitymap[0])
            print ('generating', number_of_background_samples, 'background samples took', time.time() - t0, 'seconds')
            recompute_bkg = True

        bkg_labels = bkg_samples.attributes
        bkg_score = self.suitabilityAtPoints(bkg_samples, bkg_flag = True, samplepdfonly=samplepdfonly, covweights=covweights, aggstrategy=aggstrategy)[0]


        int_samples = None
        if len(self.__samples) == 1:
            int_samples = self.__samples[0]

        if len(self.__samples) == 2:
            int_samples = self.__samples[0].mergePoints(self.__samples[1]).eliminateDuplicates()

        tr_labels = int_samples.attributes
        if self.__suitsAtsamplePoints is None:
            tr_scores = self.suitabilityAtPoints(points = int_samples, samplepdfonly=samplepdfonly, covweights=covweights, aggstrategy=aggstrategy)[0]
        else:
            tr_scores = self.__suitsAtsamplePoints[0]

        tr_y_true = np.hstack((tr_labels, bkg_labels))
        tr_y_score = np.hstack((tr_scores, bkg_score))
        #print(tr_y_true)
        #print(tr_y_score)
        #print tr_y_true.min(), tr_y_true.max()
        #print tr_y_score.min(), tr_y_score.max()
        tr_auc = evalmetric.roc_auc_score(tr_y_true, tr_y_score)
        print ('training auc =', tr_auc)
        tr_fpr, tr_tpr, tr_thresholds = evalmetric.roc_curve(tr_y_true, tr_y_score)

        t = int(time.time()*1000)
        fig = plt.figure(t)
        ax = fig.add_subplot(111)
        # plot training roc
        tr_lbl = 'training (auc=' + str(int(tr_auc*1000)/1000.0) + ')'
        plt.plot(tr_fpr, tr_tpr, '-', color = 'r', label = tr_lbl)

        # plot test roc
        te_auc = 0.0
        if test_samples is not None:
            te_labels = test_samples.attributes
            te_score = self.suitabilityAtPoints(test_samples, samplepdfonly=samplepdfonly, covweights=covweights, aggstrategy=aggstrategy)[0]
            te_y_true = np.hstack((te_labels, bkg_labels))
            te_y_score = np.hstack((te_score, bkg_score))
            te_auc = evalmetric.roc_auc_score(te_y_true, te_y_score)
            print ('test auc =', te_auc)
            te_fpr, te_tpr, te_thresholds = evalmetric.roc_curve(te_y_true, te_y_score)
            te_lbl = 'test (auc=' + str(int(te_auc*1000)/1000.0) + ')'
            plt.plot(te_fpr, te_tpr, '-', color = 'g', label = te_lbl)

        # plot random prediction as reference
        x = np.arange(0, 1.2, 0.2)
        plt.plot(x, x, '-', color = 'black', label = 'random prediction (auc=0.5)')

        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.legend(loc = 'best', frameon = False, ncol = 1)
        plt.title(title)

        if conf.BLOCK_PLOT: # if block, show plot right now
            plt.show()
        else:     # otherwise, draw plot, show them all together at the end
            plt.draw()

        if conf.SAVEFIG:
            plt.savefig('figs' + os.sep + title + '.png', dpi = 300)
            plt.close()

        return tr_auc, te_auc

    def integration(self, sample_pdf, pop_pdf, samplepdfonly, integration_strategy):
        ''' integration at the knowledge level, depending on the integration strategy: mean, minimum, or maximum
        '''
        if samplepdfonly:
            tmp_holder = []
            for pdf_s in sample_pdf:
                tmp_holder.append(pdf_s / pdf_s.max())

            if integration_strategy == 'mean':
                return np.mean(np.array(tmp_holder), axis=0)
            if integration_strategy == 'minimum':
                return np.amin(np.array(tmp_holder), axis=0)
            if integration_strategy == 'maximum':
                return np.amax(np.array(tmp_holder), axis=0)
        else:
            ## consider background distribution
            tmp_holder = []
            for pdf_s in sample_pdf:
                tmp_suit = 1.0 * np.zeros_like(pop_pdf)
                tmp_suit[pop_pdf != 0.0] = pdf_s[pop_pdf != 0.0] / pop_pdf[pop_pdf != 0.0]
                tmp_suit = 1.0 / (1.0 + np.exp(-1.0 * (tmp_suit - 1.0)))
                tmp_holder.append(tmp_suit)

            if integration_strategy == 'mean':
                return np.mean(np.array(tmp_holder), axis=0)
            if integration_strategy == 'minimum':
                return np.amin(np.array(tmp_holder), axis=0)
            if integration_strategy == 'maximum':
                return np.amax(np.array(tmp_holder), axis=0)
