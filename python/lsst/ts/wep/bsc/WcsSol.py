import numpy as np

from lsst.afw.image import VisitInfo
from lsst.obs.lsst.phosim import PhosimMapper
from lsst.geom import degrees, SpherePoint, Angle
from lsst.obs.base import createInitialSkyWcs
from lsst.afw.image import SKY

from lsst.sims.coordUtils.CameraUtils import \
    pixelCoordsFromRaDec, focalPlaneCoordsFromRaDec


class WcsSol(object):

    def __init__(self, camera=None):
        """Initialize the world coordinate system (WCS) solution class.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.camera.camera.Camera, optional
            A collection of Detectors that also supports coordinate
            transformation. (the default is None.)
        """

        # Observation meta data
        self._obs = VisitInfo()

        # LSST camera object
        if (camera is None):
            self._camera = PhosimMapper().camera
        else:
            self._camera = camera

        # Detectors
        self._detectors = list()
        self._setDetectors()

        # All WCS of detectors
        self._wcsAll = dict()

    def _setDetectors(self):
        """Set the detectors."""

        self._detectors = list(self._camera)

    def getDetectors(self):
        """Get the detectors.

        Returns
        -------
        list[lsst.afw.cameraGeom.detector.detector.Detector]
            Detectors.
        """

        return self._detectors

    def setCamera(self, camera):
        """Set the camera object.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.camera.camera.Camera
            A collection of Detectors that also supports coordinate
            transformation.
        """

        self._camera = camera
        self._setDetectors()

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

            # The flipX=True kwarg is there to get the WCS to conform to PhoSim
            # pixel coordinate conventions
            wcs = createInitialSkyWcs(self._obs, detector, flipX=True)
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

        WARNING: This method does not account for apparent motion due to
        parallax. This method is only useful for mapping positions on a
        theoretical focal plane to positions on the celestial sphere.

        Parameters
        ----------
        xPix : float, list, or numpy.ndarray
            xPix is the x pixel coordinate.
        yPix : float, list, or numpy.ndarray
            yPix is the y pixel coordinate.
        chipName : str, list, or numpy.ndarray
            chipName is the name of the chip(s) on which the pixel coordinates
            are defined. This can be an array (in which case there should be
            one chip name for each (xPix, yPix) coordinate pair), or a single
            value (in which case, all of the (xPix, yPix) points will be
            reckoned on that chip).

        Returns
        -------
        float or numpy.ndarray
            RA coordinate in degree.
        float or numpy.ndarray
            Dec coordinate in degree.
        """

        xPixArray = self._toArray(xPix)
        yPixArray = self._toArray(yPix)

        chipNameArray = self._toArray(chipName)
        if (len(chipNameArray) != len(xPixArray)):
            chipNameArray = [chipName] * len(xPixArray)

        raList = []
        decList = []
        for aXPix, aYPix, aChipName in zip(xPixArray, yPixArray, chipNameArray):
            raDec = self._wcsAll[aChipName].pixelToSky(aXPix, aYPix)
            raList.append(raDec.getRa().asDegrees())
            decList.append(raDec.getDec().asDegrees())

        if (len(raList) == 1):
            return raList[0], decList[0]
        else:
            return np.array(raList), np.array(decList)

    def _toArray(self, data):
        """Change the data to array if it is not.

        Parameters
        ----------
        data : object
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

    def pixelCoordsFromRaDec(self, ra, dec, chipName=None, epoch=2000.0,
                             includeDistortion=True):
        """Get the pixel positions (or nan if not on a chip) for objects based
        on their RA, and Dec (in degrees).

        Parameters
        ----------
        ra : float or numpy.ndarray
            ra is in degrees in the International Celestial Reference System.
        dec : float or numpy.ndarray
            dec is in degrees in the International Celestial Reference System.
        chipName : numpy.ndarray, str, None
            chipName designates the names of the chips on which the pixel
            coordinates will be reckoned. If an array, there must be as many
            chipNames as there are (RA, Dec) pairs. If a single value, all of
            the pixel coordinates will be reckoned on the same chip. If None,
            this method will calculate which chip each(RA, Dec) pair actually
            falls on, and return pixel coordinates for each (RA, Dec) pair on
            the appropriate chip. (the default is None.)
        epoch : float, optional
            epoch is the mean epoch in years of the celestial coordinate
            system. (the default is 2000.0.)
        includeDistortion : bool, optional
            If True (default), then this method will expect the true pixel
            coordinates with optical distortion included.  If False, this
            method will expect TAN_PIXEL coordinates, which are the pixel
            coordinates with estimated optical distortion removed. See the
            documentation in afw.cameraGeom for more details. (the default is
            True.)

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the x pixel coordinate
            and the second row is the y pixel coordinate.
        """

        return pixelCoordsFromRaDec(ra, dec, obs_metadata=self._obs,
                                    chipName=chipName, camera=self._camera,
                                    epoch=epoch,
                                    includeDistortion=includeDistortion)

    def focalPlaneCoordsFromRaDec(self, ra, dec, epoch=2000.0):
        """Get the focal plane coordinates for all objects in the catalog.

        Parameters
        ----------
        ra : float or numpy.ndarray
            ra is in degrees in the International Celestial Reference System.
        dec : float or numpy.ndarray
            dec is in degrees in the International Celestial Reference System.
        epoch : float, optional
            epoch is the mean epoch in years of the celestial coordinate
            system. (the default is 2000.0.)

        Returns
        -------
        numpy.ndarray
            A 2-D numpy array in which the first row is the x focal plane
            coordinate and the second row is the y focal plane coordinate
            (both in millimeters).
        """

        return focalPlaneCoordsFromRaDec(ra, dec, obs_metadata=self._obs,
                                         epoch=epoch, camera=self._camera)


if __name__ == "__main__":
    pass
