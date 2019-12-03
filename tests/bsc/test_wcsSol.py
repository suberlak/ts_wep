import numpy as np
import unittest

from lsst.obs.lsst.lsstCamMapper import LsstCamMapper

from lsst.ts.wep.bsc.WcsSol import WcsSol


class TestWcsSol(unittest.TestCase):
    """Test the ComCam class."""

    def setUp(self):

        self.wcs = WcsSol()

        self.ra = 1.0
        self.dec = 2.0
        rotSkyPos = 0.0
        self.wcs.setObsMetaData(self.ra, self.dec, rotSkyPos)

    def testGetDetectors(self):

        detectors = self.wcs.getDetectors()
        self.assertEqual(len(detectors), 201)

    def testSetAndGetCamera(self):

        lsstCam = LsstCamMapper().camera
        self.wcs.setCamera(lsstCam)

        camera = self.wcs.getCamera()
        self.assertTrue(isinstance(camera, type(lsstCam)))

        detectors = self.wcs.getDetectors()
        self.assertEqual(len(detectors), 189)

    def testSetObservationMetaData(self):

        ra = 10.0
        dec = 20.0
        rotSkyPos = 30.0
        self.wcs.setObsMetaData(ra, dec, rotSkyPos)

        obs = self.wcs.getObsMetaData()
        boreSight = obs.getBoresightRaDec()
        self.assertAlmostEqual(float(boreSight.getRa()), np.deg2rad(ra))
        self.assertAlmostEqual(float(boreSight.getDec()), np.deg2rad(dec))

        rotSkyPosInObs = float(obs.getBoresightRotAngle())
        self.assertAlmostEqual(rotSkyPosInObs, np.deg2rad(rotSkyPos))

    def testRaDecFromPixelCoordsForSingleChip(self):

        xPix = 2036.5
        yPix = 2000.5
        chipName = "R22_S11"
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)

        self.assertAlmostEqual(raByWcs, self.ra, places=2)
        self.assertAlmostEqual(decByWcs, self.dec, places=2)

    @unittest.skip
    def testRaDecFromPixelCoordsArrayForSingleChip(self):

        xPix = [2032, 2032]
        yPix = [2000, 2000]
        chipName = "R22_S11"
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)

        self.assertAlmostEqual(raByWcs[0], self.ra, places=2)
        self.assertAlmostEqual(decByWcs[0], self.dec, places=2)

    @unittest.skip
    def testRaDecFromPixelCoordsForChipArray(self):

        xPix = np.array([2032, 2032])
        yPix = np.array([2000, 2000])
        chipName = np.array(["R22_S11", "R22_S11"])
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)

        self.assertEqual(len(raByWcs), 2)
        self.assertEqual(len(decByWcs), 2)

        self.assertAlmostEqual(raByWcs[0], self.ra, places=2)
        self.assertAlmostEqual(raByWcs[1], self.ra, places=2)

    @unittest.skip
    def testPixelCoordsFromRaDecWithoutChipName(self):

        xPix, yPix = self.wcs.pixelCoordsFromRaDec(self.ra, self.dec)

        self.assertAlmostEqual(xPix, 2032, places=-1)
        self.assertAlmostEqual(yPix, 1994, places=-1)

    @unittest.skip
    def testPixelCoordsFromRaDecWithChipName(self):

        chipName = "R:2,2 S:1,1"
        xPix, yPix = self.wcs.pixelCoordsFromRaDec(self.ra, self.dec,
                                                   chipName=chipName)

        self.assertAlmostEqual(xPix, 2032, places=-1)
        self.assertAlmostEqual(yPix, 1994, places=-1)

    @unittest.skip
    def testFocalPlaneCoordsFromRaDecWithZeroRot(self):

        self.wcs.setObsMetaData(0, 0, 0)
        xInMm, yInMm = self.wcs.focalPlaneCoordsFromRaDec(0, 0)

        self.assertEqual(xInMm, 0.0)
        self.assertEqual(yInMm, 0.0)

        # 0.2 arcsec = 10 um => 1 um = 0.02 arcsec => 1 mm = 20 arcsec
        # 1 arcsec = 1/3600 degree
        xInMm, yInMm = self.wcs.focalPlaneCoordsFromRaDec(20.0/3600, 0)

        self.assertAlmostEqual(xInMm, 1.0, places=3)
        self.assertAlmostEqual(yInMm, 0.0, places=3)

    @unittest.skip
    def testFocalPlaneCoordsFromRaDecWithNonZeroRot(self):

        self.wcs.setObsMetaData(0, 0, 45)

        # 0.2 arcsec = 10 um => 1 um = 0.02 arcsec => 1 mm = 20 arcsec
        # 1 arcsec = 1/3600 degree
        xInMm, yInMm = self.wcs.focalPlaneCoordsFromRaDec(20.0/3600, 0)

        self.assertAlmostEqual(xInMm, 1/np.sqrt(2), places=3)
        self.assertAlmostEqual(yInMm, -1/np.sqrt(2), places=3)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
