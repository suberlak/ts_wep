import unittest

from lsst.ts.wep.bsc.PhoSimCam import PhoSimCam


class TestPhoSimCam(unittest.TestCase):
    """Test the PhoSimCam class."""

    def setUp(self):

        self.cam = PhoSimCam()
        self.cam.setObsMetaData(0, 0, 0)

    def testGetWfsCcdList(self):

        wfsList = self.cam.getWfsCcdList()
        self.assertEqual(len(wfsList), 201)

    def testGetWavefrontSensor(self):

        wfsData = self.cam.getWavefrontSensor()
        self.assertEqual(len(wfsData), 201)


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
