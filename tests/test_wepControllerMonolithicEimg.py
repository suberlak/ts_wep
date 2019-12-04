import os
import numpy as np
from astropy.io import fits
import shutil
import unittest

from lsst.ts.wep.cwfs.Tool import ZernikeAnnularFit
from lsst.ts.wep.CamDataCollector import CamDataCollector
from lsst.ts.wep.SourceProcessor import SourceProcessor
from lsst.ts.wep.SourceSelector import SourceSelector
from lsst.ts.wep.WfEstimator import WfEstimator
from lsst.ts.wep.WepController import WepController

from lsst.ts.wep.Utility import getModulePath, FilterType, CamType, BscDbType,\
    getConfigDir


class TestWepControllerMonolithic(unittest.TestCase):
    """Test the WepController class."""

    def setUp(self):

        self.modulePath = getModulePath()
        self.dataDir = os.path.join(self.modulePath, "tests", "tmp")
        self.butlerInput = os.path.join(self.dataDir, "input")
        self.opdDir = os.path.join(self.modulePath, "tests", "testData",
                                   "opdOutput", "9005000")

        self._makeDir(self.butlerInput)

        # Configurate the WEP components
        dataCollector = CamDataCollector(self.butlerInput)
        sourSelc = self._configSourceSelector()
        sourProc = SourceProcessor()
        wfEsti = self._configWfEstimator()

        # Instantiate the WEP controller
        self.wepCntlr = WepController(dataCollector, None, sourSelc,
                                      sourProc, wfEsti)

        # Intemediate data used in the test
        self.filter = FilterType.REF

        self.neighborStarMap = dict()
        self.starMap = dict()
        self.wavefrontSensors = dict()

        self.wfsImgMap = dict()
        self.donutMap = dict()
        self.masterDonutMap = dict()

    def _makeDir(self, directory):

        if (not os.path.exists(directory)):
            os.makedirs(directory)

    def _configSourceSelector(self):

        sourSelc = SourceSelector(CamType.PhoSim, BscDbType.LocalDbForStarFile)

        # Set the criteria of neighboring stars
        starRadiusInPixel = 63
        spacingCoefficient = 2.5
        maxNeighboringStar = 1
        sourSelc.configNbrCriteria(starRadiusInPixel, spacingCoefficient,
                                   maxNeighboringStar=maxNeighboringStar)

        # Connest the database
        dbAdress = os.path.join(self.modulePath, "tests", "testData",
                                "bsc.db3")
        sourSelc.connect(dbAdress)

        return sourSelc

    def _configWfEstimator(self):

        cwfsConfigDir = os.path.join(getConfigDir(), "cwfs")
        instDir = os.path.join(cwfsConfigDir, "instData")
        algoDir = os.path.join(cwfsConfigDir, "algo")
        wfEsti = WfEstimator(instDir, algoDir)

        # Use the comcam to calculate the LSST central raft image
        # with 1.5 mm defocal distance
        wfEsti.config(solver="exp", camType=CamType.ComCam,
                      opticalModel="offAxis", defocalDisInMm=1.5,
                      sizeInPix=160, debugLevel=0)

        return wfEsti

    def tearDown(self):

        self.wepCntlr.getSourSelc().disconnect()

        shutil.rmtree(self.dataDir)

    def testMonolithicSteps(self):
        """Do the test based on the steps defined in the child class."""

        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))

    def _steps(self):
        """Sort the order of test steps.

        Yields
        ------
        str
            Step function name.
        func
            Step function generator.
        """

        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def step1_ingestEimg(self):

        # Generate the PhoSim mapper
        self.wepCntlr.getDataCollector().genPhoSimMapper()

        intraImgFiles = os.path.join(
            getModulePath(), "tests", "testData", "phosimOutput", "realComCam",
            "repackagedFiles", "intra", "lsst_e*.fits*")
        extraImgFiles = os.path.join(
            getModulePath(), "tests", "testData", "phosimOutput", "realComCam",
            "repackagedFiles", "extra", "lsst_e*.fits*")

        self.wepCntlr.getDataCollector().ingestEimages(intraImgFiles)
        self.wepCntlr.getDataCollector().ingestEimages(extraImgFiles)

    def step2_setButlerInputsPath(self):

        self.wepCntlr.setPostIsrCcdInputs(self.butlerInput)

    def step3_getTargetStar(self):

        # Set the observation meta data
        ra = 0.0
        dec = 0.0
        rotSkyPos = 0.0
        self.wepCntlr.getSourSelc().setObsMetaData(ra, dec, rotSkyPos)

        # Set the filter
        self.wepCntlr.getSourSelc().setFilter(self.filter)

        # Get the target star by file
        skyFilePath = os.path.join(self.modulePath, "tests", "testData",
                                   "phosimOutput", "realComCam",
                                   "skyComCamInfo.txt")
        neighborStarMap, starMap, wavefrontSensors = \
            self.wepCntlr.getSourSelc().getTargetStarByFile(skyFilePath, offset=0)

        # Assign the data for the following steps to use
        self.neighborStarMap = neighborStarMap
        self.starMap = starMap
        self.wavefrontSensors = wavefrontSensors

        # Do the assertion
        self.assertEqual(len(neighborStarMap), 2)
        self.assertEqual(len(starMap), 2)
        self.assertEqual(len(wavefrontSensors), 2)

    def step4_getEimgDefocalImgMap(self):

        sensorNameList = list(self.wavefrontSensors)

        intraObsId = 9005001
        extraObsId = 9005000
        obsIdList = [intraObsId, extraObsId]

        wfsImgMap = self.wepCntlr.getEimgMapByPistonDefocal(
            sensorNameList, obsIdList)

        # Assign the data for the following steps to use
        self.wfsImgMap = wfsImgMap

        # Do the assertion
        self.assertEqual(len(wfsImgMap), 2)

    def step5_getDonutMap(self):

        self.donutMap = self.wepCntlr.getDonutMap(
            self.neighborStarMap, self.wfsImgMap, self.filter,
            doDeblending=False)

        # Do the assertion
        for sensor, donutList in self.donutMap.items():
            self.assertEqual(len(donutList), 2)

    def step6a_calcWfErr(self):

        self.donutMap = self.wepCntlr.calcWfErr(self.donutMap)

        # Do the assertion
        for sensor, donutList in self.donutMap.items():
            for donut in donutList:
                wfErr = donut.getWfErr()
                self.assertEqual(wfErr.argmax(), 2)
                self.assertGreater(wfErr.max(), 100)

        # Compare with OPD
        donutList = []
        donutList.extend(self.donutMap["R22_S11"])
        donutList.extend(self.donutMap["R22_S10"])

        for aId in range(4):
            wfErr = donutList[aId].getWfErr()
            zkOfOpd = self._getZkInNmFromOpd(aId)[3:]
            delta = np.abs(wfErr[7]-zkOfOpd[7])
            self.assertLess(delta, 5)

    def _getZkInNmFromOpd(self, opdId):

        # Get the OPD data
        opdFileName = "opd_9005000_%d.fits.gz" % opdId
        opdFitsFile = os.path.join(self.opdDir, opdFileName)
        opd = fits.getdata(opdFitsFile)

        # Get x-, y-coordinate in the OPD image
        opdSize = opd.shape[0]
        opdGrid1d = np.linspace(-1, 1, opdSize)
        opdx, opdy = np.meshgrid(opdGrid1d, opdGrid1d)

        # Fit the OPD map with Zk
        idx = (opd != 0)
        znTerms = 22
        obscuration = 0.61
        zk = ZernikeAnnularFit(opd[idx], opdx[idx], opdy[idx], znTerms,
                               obscuration)

        # Return the unit in nm (the unit in OPD is um)
        return zk * 1e3

    def step6b_calcAvgWfErrOnSglCcd(self):

        avgErrMap = dict()
        for sensor, donutList in self.donutMap.items():
            avgErr = self.wepCntlr.calcAvgWfErrOnSglCcd(donutList)
            avgErrMap[sensor] = avgErr

            # Do the assertion
            self.assertEqual(avgErr.argmax(), 2)
            self.assertGreater(avgErr.max(), 100)

        # Compare with the central OPD
        wfErrOnR22S11 = avgErrMap["R22_S11"]
        zkOfOpdOnR22S11 = self._getZkInNmFromOpd(4)[3:]
        deltaOnR22S11 = np.abs(wfErrOnR22S11[7] - zkOfOpdOnR22S11[7])
        self.assertLess(deltaOnR22S11, 5)

        wfErrOnR22S10 = avgErrMap["R22_S10"]
        zkOfOpdOnR22S10 = self._getZkInNmFromOpd(5)[3:]
        deltaOnR22S10 = np.abs(wfErrOnR22S10[7] - zkOfOpdOnR22S10[7])
        self.assertLess(deltaOnR22S10, 5)

    def step7a_genMasterDonut(self):

        self.masterDonutMap = self.wepCntlr.genMasterDonut(
            self.donutMap, zcCol=np.zeros(22))

        # Do the assertion
        self.assertEqual(len(self.masterDonutMap), 2)
        for sensor, masterDonutList in self.masterDonutMap.items():
            self.assertEqual(len(masterDonutList), 1)

    def step7b_calcWfErrOfMasterDonut(self):

        self.masterDonutMap = self.wepCntlr.calcWfErr(self.masterDonutMap)

        masterDonutList = self.masterDonutMap["R22_S11"]
        wfErr = masterDonutList[0].getWfErr()

        # Do the assertion
        self.assertEqual(wfErr.argmax(), 2)
        self.assertGreater(wfErr.max(), 100)

        # Compare with the central OPD
        zkOfOpdOnR22S11 = self._getZkInNmFromOpd(4)[3:]
        deltaOnR22S11 = np.abs(wfErr[7] - zkOfOpdOnR22S11[7])
        self.assertLess(deltaOnR22S11, 30)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
