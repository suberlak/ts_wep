from lsst.ts.wep.Utility import CamType
from lsst.ts.wep.bsc.LsstCam import LsstCam
from lsst.ts.wep.bsc.LsstFamCam import LsstFamCam
from lsst.ts.wep.bsc.ComCam import ComCam
from lsst.ts.wep.bsc.PhoSimCam import PhoSimCam


class CamFactory(object):

    @staticmethod
    def createCam(camType):
        """Create the camera object.

        Parameters
        ----------
        camType : enum 'CamType'
            Camera type.

        Returns
        -------
        LsstCam, LsstFamCam, ComCam, or PhoSimCam
            Camera object.

        Raises
        ------
        ValueError
            The camera type does not match.
        """

        if (camType == CamType.LsstCam):
            return LsstCam()
        elif (camType == CamType.LsstFamCam):
            return LsstFamCam()
        elif (camType == CamType.ComCam):
            return ComCam()
        elif (camType == CamType.PhoSim):
            return PhoSimCam()
        else:
            raise ValueError("The camera type does not match.")


if __name__ == "__main__":
    pass
