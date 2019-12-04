import unittest

from lsst.ts.wep.bsc.LsstCam import LsstCam


class TestLsstCam(unittest.TestCase):
    """Test the LsstCam class."""

    def setUp(self):

        self.cam = LsstCam()
        self.cam.setObsMetaData(0, 0, 0)

    def testGetWfsCcdList(self):

        wfsList = self.cam.getWfsCcdList()

        # Should be 8 WFS in the final
        self.assertEqual(len(wfsList), 189)

    def testGetWavefrontSensor(self):

        wfsData = self.cam.getWavefrontSensor()

        # Should be 8 WFS in the final
        self.assertEqual(len(wfsData), 189)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
