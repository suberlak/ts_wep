from lsst.obs.lsst.lsstCamMapper import LsstCamMapper
from lsst.afw.cameraGeom.detector.detector import DetectorType

from lsst.ts.wep.bsc.CameraData import CameraData


class LsstCam(CameraData):

    def __init__(self):
        """Initialize the LSST camera class."""
        super().__init__(LsstCamMapper().camera, False, [DetectorType.WAVEFRONT])


if __name__ == "__main__":
    pass
