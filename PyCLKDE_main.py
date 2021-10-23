# Author: Guiming Zhang
# Last update: Oct. 23 2021
import os, sys, time, platform, json
import numpy as np
sys.path.insert(0, 'utility')
from utility import conf, raster, points, hsm_integration, util

def main():
    ''' test driver
    '''
    # project data directory
    dataDir = 'data'
    # covariate data directory (geotiff files)
    tifDir = dataDir + os.sep + 'covariates'

    # geotiff file names
    tiffns = ["BIO1.tif", "BIO2.tif"]

    ### read in environmental data layers [standardized]
    envrasters = util.readEnvDataLayers(tifDir, tiffns, pdf=False, tile_mode=True)
    evweights = np.ones(len(envrasters)) # by default, all covariates are equally weighted
    predictors = envrasters
    ratios = evweights
    print( 'done reading environmental rasters...')

    points_bkg = points.Points() # for computing background pdf
    points_bkg.readFromCSV(dataDir + os.sep + 'localities' + os.sep + 'background_pnts.csv')
    points_bkg.plot(title = "background_pnts", radius = 0.5, color='gray')


    samplefnA = dataDir + os.sep + 'localities' + os.sep + 'empidonax_virescens_ebd.csv'
    samplefnB = dataDir + os.sep + 'localities' + os.sep + 'empidonax_virescens_inat.csv'

    ### read in species occurrences
    samplesA = points.Points()
    samplesA.readFromCSV(samplefnA)
    samplesA.plot(title = "empidonax_virescens_ebd", radius = 0.5, color='gray')

    samplesB = points.Points()
    samplesB.readFromCSV(samplefnB)
    samplesB.plot(title = "empidonax_virescens_inat", radius = 0.5, color='gray')

    samplesAll = samplesA.mergePoints(samplesB).eliminateDuplicates()
    samplesAll.plot(title = "empidonax_virescens_all", radius = 0.5, color='gray')

    ## plot pdf curves
    samplesAll.plot_density_curves(envrasters, points_bkg=points_bkg, flag = 'empidonax_virescens')

    ## random background location needed in plot roc
    bkg_samples_fn = dataDir + os.sep + 'localities' + os.sep + 'bkgnd_rnd_roc.csv'
    bkg_samples = points.Points()
    bkg_samples.readFromCSV(bkg_samples_fn)
    bkg_samples.plot(title = "bkgnd_rnd_roc", radius = 0.5, color='gray')
    print ('done reading samples...')

    ## suitability mapping
    ## modeling and mapping based on species occurrences from eBird ONLY
    hsm = hsm_integration.HSM(predictors, [samplesA], points_bkg=points_bkg, sample_pdf_update=True)
    # plot roc curve
    hsm.plot_roc_curve(test_samples = samplesA, bkg_samples = bkg_samples, title = 'ROC Curve_ebd')
    hsm.suitabilityMapTile(suitabilityfn='hs_empidonax_virescens_ebd.tif')

    ## integrating occurrences from eBird and iNaturalist at data-level
    hsm = hsm_integration.HSM(predictors, [samplesA, samplesB], integration_level = 'data', points_bkg=points_bkg, sample_pdf_update=True)
    hsm.plot_roc_curve(test_samples = samplesAll, bkg_samples = bkg_samples, title = 'ROC Curve_ebdinat_data')
    hsm.suitabilityMapTile(suitabilityfn='hs_empidonax_virescens_ebdinat_data.tif')


    ## integrating occurrences from eBird and iNaturalist at knowledge-level
    hsm = hsm_integration.HSM(predictors, [samplesA, samplesB], integration_level = 'knowledge', integration_strategy = 'minimum', points_bkg=points_bkg, sample_pdf_update=True)
    hsm.plot_roc_curve(test_samples = samplesAll, bkg_samples = bkg_samples, title = 'ROC Curve_ebdinat_knowedgemin')
    hsm.suitabilityMapTile(suitabilityfn='hs_empidonax_virescens_ebdinat_knowledgemin.tif')

    ## integrating occurrences from eBird and iNaturalist at model-level
    hsm = hsm_integration.HSM(predictors, [samplesA, samplesB], integration_level = 'model', integration_strategy = 'minimum', points_bkg=points_bkg, sample_pdf_update=True)
    hsm.plot_roc_curve(test_samples = samplesAll, bkg_samples = bkg_samples, title = 'ROC Curve_ebdinat_modelmin')
    hsm.suitabilityMapTile(suitabilityfn='hs_empidonax_virescens_ebdinat_modelmin.tif')

if __name__ == "__main__":

    main()
