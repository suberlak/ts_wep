import lsst.daf.persistence as dafPersist
from lsst.ts.wep.bsc.DonutDetector import DonutDetector
from lsst.ts.wep.bsc.LocalDatabaseForStarFile import LocalDatabaseForStarFile
from lsst.ts.wep.cwfs.TemplateUtils import createTemplateImage
from lsst.ts.wep.ParamReader import ParamReader


class LocalDatabaseFromImage(LocalDatabaseForStarFile):

    PRE_TABLE_NAME = "StarTable"

    def insertDataFromImage(self, butlerRootPath, settingFileInst,
                            visitList, defocalState,
                            filterType, wavefrontSensors, camera,
                            skiprows=1, keepFile=True,
                            fileOut='foundDonuts.txt'):

        templateType = settingFileInst.getSetting("templateType")
        donutImgSize = settingFileInst.getSetting("donutImgSizeInPixel")
        overlapDistance = 2.*settingFileInst.getSetting("starRadiusInPixel")
        skyDf = self.identifyDonuts(butlerRootPath, visitList, filterType,
                                    defocalState, wavefrontSensors, camera,
                                    templateType, donutImgSize,
                                    overlapDistance)
        self.writeSkyFile(skyDf, fileOut)
        self.insertDataByFile(fileOut, filterType, skiprows=1)

        return

    def identifyDonuts(self, butlerRootPath, visitList, filterType,
                       defocalState, wavefrontSensors, camera,
                       templateType, donutImgSize, overlapDistance):

        butler = dafPersist.Butler(butlerRootPath)
        sensorList = butler.queryMetadata('postISRCCD', 'detectorName')
        visitOn = visitList[0]
        full_unblended_df = None
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

            raw = butler.get('postISRCCD', **data_id)
            template = createTemplateImage(defocalState,
                                           detector, [[2000., 2000.]],
                                           templateType, donutImgSize)
            donut_detect = DonutDetector(template)
            donut_df = donut_detect.detectDonuts(raw, overlapDistance)
            ranked_unblended_df = donut_detect.rankUnblendedByFlux(donut_df,
                                                                   raw)
            ranked_unblended_df = ranked_unblended_df.reset_index(drop=True)

            ra, dec = camera._wcs.raDecFromPixelCoords(
                ranked_unblended_df['x_center'].values,
                ranked_unblended_df['y_center'].values,
                detector, epoch=2000.0, includeDistortion=True
            )

            ranked_unblended_df['ra'] = ra
            ranked_unblended_df['dec'] = dec

            if full_unblended_df is None:
                full_unblended_df = ranked_unblended_df.copy(deep=True)
            else:
                full_unblended_df = full_unblended_df.append(
                    ranked_unblended_df)

        full_unblended_df = full_unblended_df.reset_index(drop=True)
        # FIXME: Actually estimate magnitude
        full_unblended_df['mag'] = 15.

        return full_unblended_df

    def writeSkyFile(self, unblendedDf, fileOut):

        with open(fileOut, 'w') as file:
            file.write("# Id\t Ra\t\t Decl\t\t Mag\n")
            unblendedDf.to_csv(file, columns=['ra', 'dec', 'mag'],
                               header=False, sep='\t', float_format='%3.6f')

        return
