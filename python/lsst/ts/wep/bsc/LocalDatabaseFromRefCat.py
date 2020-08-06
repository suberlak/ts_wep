import os
import numpy as np
import lsst.daf.persistence as dafPersist
import lsst.geom
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.base as measBase
from copy import deepcopy
from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask, MagnitudeLimit
from lsst.meas.astrom import AstrometryTask
from lsst.afw.image.utils import defineFilter
from lsst.ts.wep.bsc.DonutDetector import DonutDetector
from lsst.ts.wep.bsc.LocalDatabaseFromImage import LocalDatabaseFromImage
from lsst.ts.wep.cwfs.TemplateUtils import createTemplateImage


class LocalDatabaseFromRefCat(LocalDatabaseFromImage):

    PRE_TABLE_NAME = "StarTable"

    def insertDataFromRefCat(self, butlerRootPath, settingFileInst,
                            visitList, defocalState,
                            filterType, wavefrontSensors, camera,
                            skiprows=1, keepFile=True,
                            fileOut='foundDonuts.txt'):

        centroidTemplateType = settingFileInst.getSetting("centroidTemplateType")
        donutImgSize = settingFileInst.getSetting("donutImgSizeInPixel")
        overlapDistance = settingFileInst.getSetting("minUnblendedDistance")
        maxSensorStars = settingFileInst.getSetting("maxSensorStars")
        refCatDir = settingFileInst.getSetting("refCatDir")
        refButler = dafPersist.Butler(refCatDir)
        self.refObjLoader = LoadIndexedReferenceObjectsTask(butler=refButler)
        skyDf = self.identifyDonuts(butlerRootPath, visitList, filterType,
                                    defocalState, wavefrontSensors, camera,
                                    centroidTemplateType, donutImgSize,
                                    overlapDistance, maxSensorStars)
        self.writeSkyFile(skyDf, fileOut)
        self.insertDataByFile(fileOut, filterType, skiprows=1)

        return

    def identifyDonuts(self, butlerRootPath, visitList, filterType,
                       defocalState, wavefrontSensors, camera,
                       templateType, donutImgSize, overlapDistance,
                       maxSensorStars=None):

        butler = dafPersist.Butler(butlerRootPath)
        sensorList = butler.queryMetadata('postISRCCD', 'detectorName')
        visitOn = visitList[0]
        full_ref_cat_df = None
        for detector, wavefrontSensor in wavefrontSensors.items():

            raftStr, sensorStr = detector.split(' ')
            raftDigits = raftStr.split(':')[1].split(',')
            raft = 'R%s%s' % (raftDigits[0], raftDigits[1])
            sensorDigits = sensorStr.split(':')[1].split(',')
            sensor = 'S%s%s' % (sensorDigits[0], sensorDigits[1])

            if sensor not in sensorList:
                continue

            data_id = {'visit': visitOn, 'filter': filterType.toString(),
                       'raftName': raft, 'detectorName': sensor}
            print(data_id)

            # TODO: Rename this to reflect this is postISR not raw image.
            raw = butler.get('postISRCCD', **data_id)

            template = createTemplateImage(defocalState,
                                           detector, [[2000., 2000.]],
                                           templateType, donutImgSize)
            donut_detect = DonutDetector(template)
            # min_overlap_distance = 10. # Use for detecting for ref_cat matching
            donut_df = donut_detect.detectDonuts(raw, overlapDistance)
            source_cat = self.makeSourceCat(donut_df)
            wcs_solver_result = self.solveWCS(raw, source_cat)
            ref_cat = wcs_solver_result.refCat
            ref_cat_df = ref_cat.asAstropy().to_pandas()
            x_lim, y_lim = list(raw.getDimensions())
            ref_cat_df = ref_cat_df.query(str('centroid_x < %i and centroid_y < %i and ' %
                                              (x_lim - donutImgSize, y_lim - donutImgSize) +
                                              'centroid_x > %i and centroid_y > %i' %
                                              (donutImgSize, donutImgSize)))

            ranked_ref_cat_df = ref_cat_df.sort_values(['phot_g_mean_flux'],
                                                       ascending=False)
            ranked_ref_cat_df = ranked_ref_cat_df.reset_index(drop=True)
            ranked_ref_cat_df = donut_detect.labelUnblended(ranked_ref_cat_df,
                                                            overlapDistance,
                                                            'centroid_x',
                                                            'centroid_y')
            ranked_ref_cat_df = ranked_ref_cat_df.query('blended == False').reset_index(drop=True)

            ranked_ref_cat_df['ra'] = np.degrees(ranked_ref_cat_df['coord_ra'])
            ranked_ref_cat_df['dec'] = np.degrees(ranked_ref_cat_df['coord_dec'])
            ranked_ref_cat_df['raft'] = raft
            ranked_ref_cat_df['sensor'] = sensor

            # Get magnitudes
            # Code cmmented out below will be useful when we can get a flux
            # off the image for the objects
            # photo_calib = raw.getPhotoCalib()
            # mag_list = []
            # for mean_flux in ranked_ref_cat_df['phot_g_mean_flux'].values:
            #     mag_list.append(photo_calib.instFluxToMagnitude(mean_flux))

            # Convert microJy to mags
            mag_list = 23.9 - 2.5*np.log10(1e-3*ranked_ref_cat_df['phot_g_mean_flux'].values)
            ranked_ref_cat_df['mag'] = mag_list

            bright_mag_limit = 11.
            faint_mag_limit = 15.
            ranked_ref_cat_df = \
                ranked_ref_cat_df.query('mag > %f' % bright_mag_limit).reset_index(drop=True)
            if maxSensorStars is None:
                ranked_ref_cat_df = ranked_ref_cat_df.query('mag < %f' % faint_mag_limit).reset_index(drop=True)
            else:
                ranked_ref_cat_df = ranked_ref_cat_df.iloc[:maxSensorStars]

            # Make coordinate change appropriate to sourProc.dmXY2CamXY
            # FIXME: This is a temporary workaround
            # Transpose because wepcntl. _transImgDmCoorToCamCoor
            dimY, dimX = list(raw.getDimensions())
            pixelCamX = ranked_ref_cat_df['centroid_x'].values
            pixelCamY = dimX - ranked_ref_cat_df['centroid_y'].values
            ranked_ref_cat_df['x_center'] = pixelCamX
            ranked_ref_cat_df['y_center'] = pixelCamY

            ra, dec = camera._wcs.raDecFromPixelCoords(
                ranked_ref_cat_df['x_center'].values,
                ranked_ref_cat_df['y_center'].values,
                # pixelCamX, pixelCamY,
                detector, epoch=2000.0, includeDistortion=True
            )
            print(ranked_ref_cat_df['ra'])
            ranked_ref_cat_df['ra'] = ra
            ranked_ref_cat_df['dec'] = dec
            print(ranked_ref_cat_df['ra'])

            if full_ref_cat_df is None:
                full_ref_cat_df = ranked_ref_cat_df.copy(deep=True)
            else:
                full_ref_cat_df = full_ref_cat_df.append(
                    ranked_ref_cat_df)

        full_ref_cat_df = full_ref_cat_df.reset_index(drop=True)

        # TODO: Comment out when not debugging
        full_ref_cat_df.to_csv('image_donut_df.csv')

        return full_ref_cat_df

    def makeSourceCat(self, donutDf):
        """Make a source catalog by reading the position reference stars and distorting the positions
        """

        sourceSchema = afwTable.SourceTable.makeMinimalSchema()
        measBase.SingleFrameMeasurementTask(schema=sourceSchema)

        sourceCat = afwTable.SourceCatalog(sourceSchema)
        sourceCentroidKey = afwTable.Point2DKey(sourceSchema["slot_Centroid"])
        sourceIdKey = sourceSchema["id"].asKey()
        sourceInstFluxKey = sourceSchema["slot_ApFlux_instFlux"].asKey()
        sourceInstFluxErrKey = sourceSchema["slot_ApFlux_instFluxErr"].asKey()

        #populate source catalog with objects from reference catalog
        sourceCat.reserve(len(donutDf['x_center']))

        for i,(_x,_y) in enumerate(zip(donutDf['x_center'], donutDf['y_center'])):
            src = sourceCat.addNew()
            src.set(sourceIdKey, i)

            src.set(sourceCentroidKey, lsst.geom.Point2D(_x, _y))
            src.set(sourceInstFluxKey, 15.)
            src.set(sourceInstFluxErrKey, 15./100)

        return sourceCat

    def solveWCS(self, raw, sourceCat):

        exposure = deepcopy(raw)

        astromConfig = AstrometryTask.ConfigClass()

        magLimit = MagnitudeLimit()
        magLimit.minimum = 11
        magLimit.maximum = 12
        astromConfig.referenceSelector.magLimit = magLimit
        astromConfig.referenceSelector.magLimit.fluxField = "phot_rp_mean_flux"
        astromConfig.matcher.minMatchedPairs = 4
        astromConfig.matcher.maxRotationDeg = 2.99
        astromConfig.matcher.maxOffsetPix = 40
        astromConfig.wcsFitter.order = 3
        astromConfig.wcsFitter.numRejIter = 0
        astromConfig.wcsFitter.maxScatterArcsec = 15

        # this is a bit sleazy (as RHL would say) but I'm just forcing the exposure
        # to have the same name as the one in the Gaia catalog for now
        referenceFilterName = 'phot_rp_mean'
        defineFilter(referenceFilterName, 656.28)
        referenceFilter = afwImage.filter.Filter(referenceFilterName)
        exposure.setFilter(referenceFilter)

        sourceSchema = sourceCat.getSchema()
        solver = AstrometryTask(config=astromConfig, refObjLoader=self.refObjLoader, schema=sourceSchema,)
        results = solver.run(sourceCat=sourceCat, exposure=exposure,)

        return results