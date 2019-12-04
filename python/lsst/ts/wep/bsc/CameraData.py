import numpy as np
import copy

from lsst.ts.wep.bsc.WcsSol import WcsSol
from lsst.ts.wep.Utility import FilterType


class CameraData(object):

    def __init__(self, camera, flipX):
        """Initialize the camera data class.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.camera.camera.Camera
            A collection of Detectors that also supports coordinate
            transformation.
        flipX : bool
            If False, +X is along W, if True +X is along E.
        """

        self._wcs = WcsSol(camera, flipX)

        # List of wavefront sensor CCD name
        self._wfsCcd = []

        # Dictionary of (x, y) coordinates of detector corners and dimensions
        # of detection. The dictonary key is the ccd name.
        self._corners = dict()
        self._dimension = dict()

        self._initDetectors()

    def _initDetectors(self):
        """Initializes the camera detectors."""

        for detector in self._wcs.getDetectors():

            # Collect the ccd name
            detectorName = detector.getName()
            self._wfsCcd.append(detectorName)

            # Get the detector corners
            bbox = detector.getBBox()
            xmin = bbox.getMinX()
            xmax = bbox.getMaxX()
            ymin = bbox.getMinY()
            ymax = bbox.getMaxY()
            self._corners[detectorName] = \
                (np.array([xmin, xmin, xmax, xmax]),
                 np.array([ymin, ymax, ymin, ymax]))

            # The CCD dimension here is an estimation.
            # Based on LCA-13381, there are three types of sensors.
            # e2V CCD250: 40.04 mm x 40.96 mm
            # STA 4400: 20.00 mm x 40.72 mm
            # STA 3800C: 40.00 mm x 40.72 mm

            dim1, dim2 = bbox.getDimensions()
            self._dimension[detectorName] = (int(dim1), int(dim2))

    def getWfsCcdList(self):
        """Get the list of wavefront sensor CCD list.

        Returns
        -------
        list
            Wavefront sensor CCD list.
        """

        return self._wfsCcd

    def getWfsCorner(self, detectorName):
        """Get the wavefront sensor corner.

        Parameters
        ----------
        detectorName : str
            Detector Name (e.g. "R:2,2 S:1,1").

        Returns
        -------
        tuple
            Wavefront sensor corner.
        """

        return self._corners[detectorName]

    def getCcdDim(self, detectorName):
        """Get the CCD dimension.

        Parameters
        ----------
        detectorName : str
            Detector name (e.g. "R:2,2 S:1,1").

        Returns
        -------
        tuple
            CCD dimension in pixel.
        """

        return self._dimension[detectorName]

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

        self._wcs.setObsMetaData(ra, dec, rotSkyPos)

    def populatePixelFromRADecl(self, stars):
        """Populates the RAInPixel and DeclInPixel coordinates to the stars.

        Parameters
        ----------
        stars : StarData
            The stars to populate.

        Returns
        -------
        StarData
            The stars with x-, y-pixel data populated.
        """

        # Do the shallow copy
        populatedStar = copy.copy(stars)

        # Populate the pixel data
        ra = populatedStar.getRA()
        decl = populatedStar.getDecl()
        chipName = np.array([populatedStar.getDetector()] * len(ra))
        raInPixel, declInPixel = self._wcs.pixelCoordsFromRaDec(
            ra, decl, chipName)

        populatedStar.setRaInPixel(raInPixel)
        populatedStar.setDeclInPixel(declInPixel)

        return populatedStar

    def getStarsOnDetector(self, stars, offset):
        """Get the stars on the detector according to the pixel position.

        Parameters
        ----------
        stars : StarData
            Star information.
        offset : float
            Offset to the dimension of camera. If the detector dimension is 10
            (assume 1-D), the star's position between -offset and 10+offset
            will be seem to be on the detector.

        Returns
        -------
        StarData
            The stars on the detector.
        """

        # Do the shallow copy
        starsOnDet = copy.copy(stars)

        # Get the index that will keep the data
        starsRaInPixel = starsOnDet.getRaInPixel()
        starsDeclInPixel = starsOnDet.getDeclInPixel()
        ccdDim = self.getCcdDim(starsOnDet.getDetector())

        keep = []
        for ii in range(len(starsRaInPixel)):
            if (-offset <= starsRaInPixel[ii] <= ccdDim[0] + offset and
               -offset <= starsDeclInPixel[ii] <= ccdDim[1] + offset):
                keep.append(ii)

        # Remove the stars that are not on the detector
        starsOnDet.setId(self._getKeepItem(starsOnDet.getId(), keep))
        starsOnDet.setRA(self._getKeepItem(starsOnDet.getRA(), keep))
        starsOnDet.setDecl(self._getKeepItem(starsOnDet.getDecl(), keep))
        starsOnDet.setRaInPixel(self._getKeepItem(starsRaInPixel, keep))
        starsOnDet.setDeclInPixel(self._getKeepItem(starsDeclInPixel, keep))

        for filterType in FilterType:

            if (filterType != FilterType.REF):
                magArray = starsOnDet.getMag(filterType)

                if (len(magArray) != 0):
                    starsOnDet.setMag(filterType, self._getKeepItem(magArray,
                                                                    keep))

        return starsOnDet

    def _getKeepItem(self, valArray, keep):
        """Get the keep items in array.

        Parameters
        ----------
        valArray : list or 1-D numpy.ndarray
            Value array.
        keep : list[int]
            List of keep index.

        Returns
        -------
        list
            The array that has only the keep values.
        """

        return [valArray[idx] for idx in keep]

    def getWavefrontSensor(self):
        """

        Get the corners of LSST curvature wavefront sensors in (ra, dec) based
        on the wavefront sensor list.

        Returns:
            dict -- (ra, dec) of four corners of each sensor with the name
            of sensor as a list. The dictionary key is the sensor name.
        """

        return self._getDetectorRaDec(self._wfsCcd)

    def _getDetectorRaDec(self, chipNames):
        """Get the (ra, dec) of CCD corners in the detector list.

        Parameters
        ----------
        chipNames : list
            List of chip name. For example, ["R22_S11", "R22_S01"].

        Returns
        -------
        dict
            This method returns a dict of list.  The dict is keyed on the name
            of the wavefront sensor.  The list contains the (RA, Dec)
            coordinates of the corners of that sensor (RA, Dec are paired as
            tuples). For example, output['R22_S11'] = [(23.0, -5.0),
            (23.1, -5.0), (23.0, -5.1), (23.1, -5.1)] would mean that the
            wavefront sensor named 'R22_S11' has its corners at RA 23,
            Dec -5; RA 23.1, Dec -5; RA 23, Dec -5.1; and RA 23.1, Dec -5.1
            Coordinates are in degrees.
        """

        ra_dec_out = dict()
        for chipName in chipNames:

            coord = self._corners[chipName]
            xPix = coord[0]
            yPix = coord[1]

            ra, dec = self._wcs.raDecFromPixelCoords(xPix, yPix, chipName)

            ra_dec_out[chipName] = [(ra[0], dec[0]), (ra[1], dec[1]),
                                    (ra[2], dec[2]), (ra[3], dec[3])]

        return ra_dec_out


if __name__ == "__main__":
    pass
