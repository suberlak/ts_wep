from lsst.obs.lsst.phosim import PhosimMapper
from lsst.afw.cameraGeom.detector.detector import DetectorType

from lsst.ts.wep.bsc.CameraData import CameraData


class PhoSimCam(CameraData):

    def __init__(self):
        """Initialize the PhoSim camera class."""

        # the flipX=True is there to get the WCS to conform to PhoSim pixel
        # coordinate conventions
        super().__init__(PhosimMapper().camera, True,
                         [DetectorType.SCIENCE, DetectorType.WAVEFRONT])


if __name__ == "__main__":
    pass
