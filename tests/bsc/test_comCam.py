import unittest

from lsst.ts.wep.bsc.StarData import StarData
from lsst.ts.wep.bsc.ComCam import ComCam
from lsst.ts.wep.Utility import FilterType


class TestComCam(unittest.TestCase):
    """Test the ComCam class and functions of parent class."""

    def setUp(self):

        # Boresight (unit: degree)
        ra = 0.0    # 0 <= RA <= 360
        dec = 30.0   # -90 <= Dec <= 90
        rotSkyPos = 0.0
        self.camera = ComCam()
        self.camera.setObsMetaData(ra, dec, rotSkyPos)

        self.stars = StarData([123, 456, 789], [0.1, 0.2, 0.3],
                              [2.1, 2.2, 2.3], [2.0, 3.0, 4.0],
                              [2.1, 2.1, 4.1], [2.2, 3.2, 4.2],
                              [2.3, 3.3, 4.3], [], [])

    def testGetWfsCcdList(self):

        wfsCcdList = self.camera.getWfsCcdList()

        self.assertEqual(len(wfsCcdList), 9)
        self.assertTrue("R22_S11" in wfsCcdList)
        self.assertFalse("R21_S11" in wfsCcdList)

    def testGetWfsCorner(self):

        wfsCorner = self.camera.getWfsCorner("R22_S11")

        self.assertEqual(len(wfsCorner), 2)
        self.assertRaises(KeyError, self.camera.getWfsCorner, "R21_S11")

    def testGetCcdDim(self):

        ccdDim = self.camera.getCcdDim("R22_S11")

        self.assertEqual(ccdDim, (4072, 4000))

    def testGetWavefrontSensor(self):

        wfsData = self.camera.getWavefrontSensor()
        self.assertEqual(len(wfsData), 9)

    def testPopulatePixelFromRADecl(self):

        self.assertEqual(len(self.stars.getRaInPixel()), 0)

        populatedStar = self._populatePixelFromRADecl()

        self.assertEqual(len(self.stars.getRaInPixel()), 0)
        self.assertEqual(len(self.stars.getDeclInPixel()), 0)
        self.assertEqual(len(populatedStar.getRaInPixel()), 3)
        self.assertEqual(len(populatedStar.getDeclInPixel()), 3)
        self.assertNotEqual(id(populatedStar), id(self.stars))

    def _populatePixelFromRADecl(self):

        self.stars.setDetector("R22_S11")
        populatedStar = self.camera.populatePixelFromRADecl(self.stars)

        return populatedStar

    def testRemoveStarsNotOnDetectorWithLargeOffset(self):

        stars = self._populatePixelFromRADecl()
        starsOnDet = self.camera.getStarsOnDetector(stars, 1e7)

        self.assertEqual(len(starsOnDet.getId()), 3)
        self.assertEqual(len(starsOnDet.getRA()), 3)
        self.assertEqual(len(starsOnDet.getDecl()), 3)
        self.assertEqual(len(starsOnDet.getRaInPixel()), 3)
        self.assertEqual(len(starsOnDet.getDeclInPixel()), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.U)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.G)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.R)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.I)), 3)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Z)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Y)), 0)

        self.assertNotEqual(id(starsOnDet), id(stars))

    def testRemoveStarsNotOnDetectorWithZeroOffset(self):

        stars = self._populatePixelFromRADecl()
        starsOnDet = self.camera.getStarsOnDetector(stars, 0)

        self.assertEqual(len(starsOnDet.getId()), 0)
        self.assertEqual(len(starsOnDet.getRA()), 0)
        self.assertEqual(len(starsOnDet.getDecl()), 0)
        self.assertEqual(len(starsOnDet.getRaInPixel()), 0)
        self.assertEqual(len(starsOnDet.getDeclInPixel()), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.U)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.G)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.R)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.I)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Z)), 0)
        self.assertEqual(len(starsOnDet.getMag(FilterType.Y)), 0)

        self.assertEqual(len(stars.getId()), 3)
        self.assertEqual(len(stars.getRA()), 3)
        self.assertEqual(len(stars.getDecl()), 3)
        self.assertEqual(len(stars.getRaInPixel()), 3)
        self.assertEqual(len(stars.getDeclInPixel()), 3)
        self.assertEqual(len(stars.getMag(FilterType.U)), 3)
        self.assertEqual(len(stars.getMag(FilterType.G)), 3)
        self.assertEqual(len(stars.getMag(FilterType.R)), 3)
        self.assertEqual(len(stars.getMag(FilterType.I)), 3)
        self.assertEqual(len(stars.getMag(FilterType.Z)), 0)
        self.assertEqual(len(stars.getMag(FilterType.Y)), 0)

        self.assertNotEqual(id(starsOnDet), id(stars))


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
