import numpy as np
import unittest

from lsst.obs.lsst.phosim import PhosimMapper

from lsst.ts.wep.bsc.WcsSol import WcsSol


class TestWcsSol(unittest.TestCase):
    """Test the ComCam class."""

    def setUp(self):

        self.wcs = WcsSol(PhosimMapper().camera, True)

        self.ra = 1.0
        self.dec = 2.0
        rotSkyPos = 0.0
        self.wcs.setObsMetaData(self.ra, self.dec, rotSkyPos)

    def testGetDetectors(self):

        detectors = self.wcs.getDetectors()
        self.assertEqual(len(detectors), 201)

    def testGetCamera(self):

        camera = self.wcs.getCamera()
        self.assertTrue(isinstance(camera, type(PhosimMapper().camera)))

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

        self.assertAlmostEqual(raByWcs, self.ra)
        self.assertAlmostEqual(decByWcs, self.dec)

    def testRaDecFromPixelCoordsArrayForSingleChip(self):

        xPix = [2036.5, 2036.5]
        yPix = [2000.5, 2000.5]
        chipName = "R22_S11"
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)

        self.assertEqual(len(raByWcs), len(xPix))
        self.assertEqual(len(decByWcs), len(yPix))

        self.assertAlmostEqual(raByWcs[0], self.ra)
        self.assertAlmostEqual(decByWcs[0], self.dec)

    def testRaDecFromPixelCoordsForChipArray(self):

        xPix = [2036.5, 2036.5]
        yPix = [2000.5, 2000.5]
        chipName = np.array(["R22_S11", "R22_S11"])
        raByWcs, decByWcs = self.wcs.raDecFromPixelCoords(xPix, yPix, chipName)

        self.assertEqual(len(raByWcs), len(xPix))
        self.assertEqual(len(decByWcs), len(yPix))

        self.assertAlmostEqual(raByWcs[0], self.ra)
        self.assertAlmostEqual(raByWcs[1], self.ra)

    def testPixelCoordsFromRaDecForSingleChip(self):

        chipName = "R22_S11"
        xPix, yPix = self.wcs.pixelCoordsFromRaDec(self.ra, self.dec, chipName)

        self.assertAlmostEqual(xPix, 2036.5)
        self.assertAlmostEqual(yPix, 2000.5)

    def testPixelCoordsFromRaDecArrayForSingleChip(self):

        chipName = "R22_S11"
        lenOfRaDec = 2
        xPix, yPix = self.wcs.pixelCoordsFromRaDec(
            [self.ra]*lenOfRaDec, [self.dec]*lenOfRaDec, chipName)

        self.assertEqual(len(xPix), lenOfRaDec)
        self.assertEqual(len(yPix), lenOfRaDec)

        self.assertAlmostEqual(xPix[0], 2036.5)
        self.assertAlmostEqual(yPix[0], 2000.5)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
