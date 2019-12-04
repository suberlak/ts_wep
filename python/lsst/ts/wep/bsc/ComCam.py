from lsst.obs.lsst.comCam import LsstComCamMapper
from lsst.afw.cameraGeom.detector.detector import DetectorType

from lsst.ts.wep.bsc.CameraData import CameraData


class ComCam(CameraData):

    def __init__(self):
        """Initialize the commissioning camera class."""
        super().__init__(LsstComCamMapper().camera, False,
                         [DetectorType.SCIENCE])


if __name__ == "__main__":
    pass
