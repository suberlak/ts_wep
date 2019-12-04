from lsst.obs.lsst.lsstCamMapper import LsstCamMapper

from lsst.ts.wep.bsc.CameraData import CameraData


class LsstCam(CameraData):

    def __init__(self):
        """Initialize the LSST camera class."""
        super().__init__(LsstCamMapper().camera, False)

        # Not WFS support at w_2019_38 yet.


if __name__ == "__main__":
    pass
