from lsst.obs.lsst.lsstCamMapper import LsstCamMapper

from lsst.ts.wep.bsc.CameraData import CameraData


class LsstFamCam(CameraData):

    def __init__(self):
        """Initialize the LSST full-array mode (FAM) camera class."""
        super().__init__(LsstCamMapper().camera, False)


if __name__ == "__main__":
    pass
