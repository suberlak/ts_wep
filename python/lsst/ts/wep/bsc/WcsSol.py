import numpy as np

from lsst.afw.image import VisitInfo
from lsst.geom import degrees, SpherePoint, Angle
from lsst.obs.base import createInitialSkyWcs
from lsst.afw.image import SKY


class WcsSol(object):

    def __init__(self, camera, flipX):
        """Initialize the world coordinate system (WCS) solution class.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.camera.camera.Camera
            A collection of Detectors that also supports coordinate
            transformation.
        flipX : bool
            If False, +X is along W, if True +X is along E.
        """

        # Observation meta data
        self._obs = VisitInfo()

        # Camera object
        self._camera = camera

        # Flip the x-axis or not
        self._flipX = flipX

        # Detectors
        self._detectors = list(self._camera)

        # All WCS of detectors
        self._wcsAll = dict()

    def getDetectors(self):
        """Get the detectors.

        Returns
        -------
        list[lsst.afw.cameraGeom.detector.detector.Detector]
            Detectors.
        """

        return self._detectors

    def getCamera(self):
        """Get the camera object.

        Returns
        -------
        lsst.afw.cameraGeom.camera.camera.Camera
            A collection of Detectors that also supports coordinate
            transformation.
        """

        return self._camera

    def setObsMetaData(self, ra, dec, rotSkyPos):
        """Set the observation meta data.

        Parameters
        ----------
        ra : float
            Pointing ra in degree.
        dec : float
            Pointing decl in degree.
        rotSkyPos : float
            The orientation of the telescope in degrees.
        """

        boreSight = SpherePoint(ra*degrees, dec*degrees)
        rotSkyAng = Angle(rotSkyPos*degrees)
        self._obs = VisitInfo(boresightRaDec=boreSight,
                              boresightRotAngle=rotSkyAng,
                              rotType=SKY)

        self._wcsAll = dict()
        for detector in self._detectors:
            detectorName = detector.getName()
            wcs = createInitialSkyWcs(self._obs, detector, flipX=self._flipX)
            self._wcsAll[detectorName] = wcs

    def getObsMetaData(self):
        """Get the observation meta data.

        Returns
        -------
        lsst.afw.image.VisitInfo
            Observation meta data.
        """

        return self._obs

    def raDecFromPixelCoords(self, xPix, yPix, chipName):
        """Convert pixel coordinates into RA, Dec.

        Parameters
        ----------
        xPix : float, list, or numpy.ndarray
            X pixel coordinate.
        yPix : float, list, or numpy.ndarray
            Y pixel coordinate.
        chipName : str, list, or numpy.ndarray
            Name of the chip(s) on which the pixel coordinates are defined.
            This can be an array (in which case there should be one chip name
            for each (xPix, yPix) coordinate pair), or a single value (in
            which case, all of the (xPix, yPix) points will be reckoned on
            that chip).

        Returns
        -------
        float or numpy.ndarray
            RA coordinate in degree.
        float or numpy.ndarray
            Dec coordinate in degree.
        """

        return self._transCoord("pixelToSky", xPix, yPix, chipName)

    def _transCoord(self, func, dim1, dim2, chipName):
        """Do the coordinate transformation.

        Parameters
        ----------
        func : str
            Function to use ("pixelToSky" or "skyToPixel").
        dim1 : float, list, or numpy.ndarray
            Dimension 1.
        dim2 : float, list, or numpy.ndarray
            Dimension 2.
        chipName : str, list, or numpy.ndarray
            Name of the chip(s) on which the pixel coordinates are defined.
            This can be an array (in which case there should be one chip name
            for each (dim1, dim2) coordinate pair), or a single value (in
            which case, all of the (dim1, dim2) points will be reckoned on
            that chip).

        Returns
        -------
        float or numpy.ndarray
            Transformed dimension 1.
        float or numpy.ndarray
            Transformed dimension 2.

        Raises
        ------
        ValueError
            Function is not supported.
        """

        dim1Array = self._toArray(dim1)
        dim2Array = self._toArray(dim2)

        chipNameArray = self._toArray(chipName)
        lenOfDim1Array = len(dim1Array)
        if (len(chipNameArray) != lenOfDim1Array):
            chipNameArray = [chipName] * lenOfDim1Array

        dim1ArrayTrans = []
        dim2ArrayTrans = []
        for aDim1, aDim2, aChipName in zip(dim1Array, dim2Array, chipNameArray):

            wcsChip = self._wcsAll[aChipName]
            if (func == "pixelToSky"):
                raDec = wcsChip.pixelToSky(aDim1, aDim2)
                dim1ArrayTrans.append(raDec.getRa().asDegrees())
                dim2ArrayTrans.append(raDec.getDec().asDegrees())
            elif (func == "skyToPixel"):
                raDec = SpherePoint(aDim1*degrees, aDim2*degrees)
                pixels = wcsChip.skyToPixel(raDec)
                dim1ArrayTrans.append(pixels.getX())
                dim2ArrayTrans.append(pixels.getY())
            else:
                raise ValueError("Function '%s' is not supported." % func)

        if (lenOfDim1Array == 1):
            return dim1ArrayTrans[0], dim2ArrayTrans[0]
        else:
            return np.array(dim1ArrayTrans), np.array(dim2ArrayTrans)

    def _toArray(self, data):
        """Convert the data to array.

        Parameters
        ----------
        data : str, int, float, list, or numpy.ndarray
            Data.

        Returns
        -------
        list
            Data array.
        """

        if (isinstance(data, str)):
            return [data]

        try:
            iter(data)
            return data
        except TypeError:
            return [data]

    def pixelCoordsFromRaDec(self, ra, dec, chipName):
        """Get the pixel positions for objects based on their RA and Dec.

        Parameters
        ----------
        ra : float or numpy.ndarray
            RA in degree.
        dec : float or numpy.ndarray
            Dec in degree.
        chipName : str, list, or numpy.ndarray
            Name of the chip(s) on which the pixel coordinates are defined.
            This can be an array (in which case there should be one chip name
            for each (ra, dec) coordinate pair), or a single value (in which
            case, all of the (ra, dec) points will be reckoned on that chip).

        Returns
        -------
        float or numpy.ndarray
            X pixel coordinate.
        float or numpy.ndarray
            Y pixel coordinate.
        """

        return self._transCoord("skyToPixel", ra, dec, chipName)


if __name__ == "__main__":
    pass
