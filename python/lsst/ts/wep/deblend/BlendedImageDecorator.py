import numpy as np

from scipy.ndimage.morphology import binary_opening, binary_closing, \
    binary_erosion
from scipy.ndimage.interpolation import shift
from scipy.optimize import minimize_scalar

from lsst.ts.wep.deblend.AdapThresImage import AdapThresImage
from lsst.ts.wep.deblend.nelderMeadModify import nelderMeadModify

from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from lsst.afw.image import ImageF
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve, correlate2d


class BlendedImageDecorator():

    def __init__(self):
        """Initialize the BlendedImageDecorator class."""

        self.__image = AdapThresImage()

    def __getattr__(self, attributeName):
        """Use the functions and attributes hold by the object.

        Parameters
        ----------
        attributeName : str
            Name of attribute or function.

        Returns
        -------
        object
            Returned values.
        """

        return getattr(self.__image, attributeName)

    def newCentroidFinder(self, imageBinary, templateImgBinary,
                          n_donuts):
        """New Version of Centroid Finder

        Parameters
        ----------
        imageBinary: numpy.ndarray
            Binary image of postage stamp

        templateImgBinary: numpy.ndarray
            Binary image of template donut

        Returns
        -------
        nx, ny: float
            x, y pixel coordinates for donut centroid
        """

        temp_convolve = convolve(imageBinary, templateImgBinary, mode='constant', cval=0.0)
        ranked_convolve = np.argsort(temp_convolve.flatten())[::-1]
        cutoff = len(np.where(temp_convolve.flatten() > 0.95*np.max(temp_convolve))[0])
        ranked_convolve = ranked_convolve[:cutoff]
        nx, ny = np.unravel_index(ranked_convolve, np.shape(imageBinary))

        kmeans = KMeans(n_clusters=n_donuts).fit(np.array([ny, nx]).T)
        labels = kmeans.labels_

        cent_x = []
        cent_y = []

        for label_num in range(n_donuts):
            nx_label, ny_label = np.unravel_index(ranked_convolve[labels == label_num][0],
                                                  np.shape(imageBinary))
            cent_x.append(nx_label)
            cent_y.append(ny_label)

        return cent_x, cent_y

    def deblendDonut(self, iniGuessXY, n_donuts, sensor_name,
                     defocal_type, new_centroid):
        """
        Get the deblended donut image.

        Parameters
        ----------
        iniGuessXY : tuple or list
            Initial guess of (x, y) position of neighboring star.

        Returns
        -------
        numpy.ndarray
            Deblended donut image.
        float
            Position x in pixel.
        float
            Position y in pixel.
        """

        # Deblended image
        imgDeblend = []

        # Postion of centroid

        # Get the initial guess of brightest donut
        if new_centroid is False:
            centroidFind = self.getCentroidFind(sensor_name, defocal_type)
            imgBinary = centroidFind.getImgBinary(self.getImg())
            realcx, realcy, realR = centroidFind.getCenterAndRfromImgBinary(
                imgBinary)

            # Remove the salt and pepper noise noise of resImgBinary
            imgBinary = binary_opening(imgBinary).astype(float)
            imgBinary = binary_closing(imgBinary).astype(float)

            # Check the image quality
            if (not realcx):
                return imgDeblend, realcx, realcy

            # Get the binary image by adaptive threshold
            adapcx, adapcy, adapR, adapImgBinary = self.getCenterAndR_adap()
        else:
            templateArray = self.deblendUtils.createTemplateImage(sensor_name,
                                                                  defocal_type)

            templateImage = AdapThresImage()
            templateImage.setImg(image=templateArray)
            templatecx, templatecy, templateR, templateImgBinary = \
                templateImage.getCenterAndR_adap()
            templateImgBinary = binary_closing(templateImgBinary)

            adapcx, adapcy, adapR, adapImgBinary = self.getCenterAndR_adap()
            cx_list, cy_list = self.newCentroidFinder(adapImgBinary,
                                                      templateImgBinary,
                                                      n_donuts)
            # Order the centroids to figure out which is neighbor star
            centroid_dist = cdist(np.array([iniGuessXY]),
                                  np.array([cx_list, cy_list]).T)
            iniGuess_dist_order = np.argsort(centroid_dist[0])
            # Update coords of neighbor star and bright star with centroid pos
            realcx = cx_list[iniGuess_dist_order[1]]
            realcy = cy_list[iniGuess_dist_order[1]]
            iniGuessXY = [cx_list[iniGuess_dist_order[0]],
                          cy_list[iniGuess_dist_order[0]]]

            imgBinary = np.zeros(np.shape(adapImgBinary))
            imgBinary[:np.shape(templateImgBinary)[0],
                      :np.shape(templateImgBinary)[1]] += templateImgBinary

            xBin = int(realcx - templatecx)
            yBin = int(realcy - templatecy)

            imgBinary = shift(imgBinary, [xBin, yBin])
            imgBinary[imgBinary < 0] = 0

        # Calculate the system error by only taking the background signal
        bg1D = self.getImg().flatten()
        bgImgBinary1D = adapImgBinary.flatten()
        background = bg1D[bgImgBinary1D == 0]
        bgPhist, binEdges = np.histogram(background, bins=256)
        sysError = np.mean(binEdges[0:2])

        # Remove the system error
        noSysErrImage = self.getImg() - sysError
        noSysErrImage[noSysErrImage < 0] = 0

        # Get the residure map
        resImgBinary = adapImgBinary - imgBinary

        # Compensate the zero element for subtraction
        resImgBinary[np.where(resImgBinary < 0)] = 0

        # Remove the salt and pepper noise noise of resImgBinary
        resImgBinary = binary_opening(resImgBinary).astype(float)

        # Calculate the shifts of x and y
        x0 = int(iniGuessXY[0] - realcx)
        y0 = int(iniGuessXY[1] - realcy)
        if new_template is not None:
            neighbor_opt_array = np.array([y0, x0])
        else:
            neighbor_opt_array = np.array([x0, y0])

        xoptNeighbor = nelderMeadModify(self._funcResidue, neighbor_opt_array,
                                        args=(imgBinary, resImgBinary),
                                        step=15)

        # Shift the main donut image to fitted position of neighboring star
        fitImgBinary = shift(imgBinary, [int(xoptNeighbor[0][1]),
                                         int(xoptNeighbor[0][0])])

        # Handle the numerical error of shift. Regenerate a binary image.
        fitImgBinary[fitImgBinary > 0.5] = 1
        fitImgBinary[fitImgBinary < 0.5] = 0

        # Get the overlap region between main donut and neighboring donut
        imgOverlapBinary = imgBinary + fitImgBinary
        imgOverlapBinary[imgOverlapBinary < 1.5] = 0
        imgOverlapBinary[imgOverlapBinary > 1.5] = 1

        # Get the overall binary image
        imgAllBinary = imgBinary + fitImgBinary
        imgAllBinary[imgAllBinary > 1] = 1

        # Get the reference image for the fitting
        imgRef = noSysErrImage*imgAllBinary

        # Calculate the magnitude ratio of image
        imgMainDonut = noSysErrImage*imgBinary
        imgFit = shift(imgMainDonut, [int(xoptNeighbor[0][1]),
                                      int(xoptNeighbor[0][0])])

        xoptMagNeighbor = minimize_scalar(
            self._funcMag, bounds=(0, 1), method="bounded",
            args=(imgMainDonut, imgOverlapBinary, imgFit, imgRef, xoptNeighbor[0]))

        imgDeblend = imgMainDonut - xoptMagNeighbor.x*imgFit*imgOverlapBinary

        # Repair the boundary of image
        imgDeblend = self._repairBoundary(imgOverlapBinary, imgBinary,
                                          imgDeblend)

        # Calculate the centroid position of donut
        realcy, realcx = center_of_mass(imgBinary)

        return imgDeblend, realcx, realcy

    def _repairBoundary(self, imgOverlapBinary, imgBinary, imgDeblend):
        """Compensate the values on boundary of overlap region between the
        bright star and neighboring star.

        Parameters
        ----------
        imgOverlapBinary : numpy.ndarray[int]
            Binary impage of overlap between bright star and neighboring star.
        imgBinary : numpy.ndarray[int]
            Binary image of bright star.
        imgDeblend : numpy.ndarray[float]
            Deblended donut image.

        Returns
        -------
        numpy.ndarray[float]
            Repaired deblended donut image.
        """

        # Copy the original data
        repairImgDeblend = imgDeblend.copy()

        # Get the boundary of overlap region
        boundaryOverlap = imgOverlapBinary - binary_erosion(imgOverlapBinary)

        # Find all boundary points
        m, n = np.where(boundaryOverlap == 1)

        for ii in range(len(m)):

            # Correct values that are not on the boundary next to environment
            if (imgBinary[m[ii]-1:m[ii]+2, n[ii]-1:n[ii]+2].all()):

                # Modify the value in column
                neighborValues = repairImgDeblend[m[ii], n[ii]-4:n[ii]+5]
                temp = neighborValues[neighborValues != 0]
                stdTemp = np.std(temp)
                meanTemp = np.mean(temp)

                for kk in range(9):
                    testValue = repairImgDeblend[m[ii], n[ii]-4+kk]
                    if (testValue != 0):
                        if (testValue >= meanTemp + 2*stdTemp) or \
                           (testValue <= meanTemp - 2*stdTemp):
                            repairImgDeblend[m[ii], n[ii]-4+kk] = \
                                (repairImgDeblend[m[ii], n[ii]-5+kk] +
                                 repairImgDeblend[m[ii], n[ii]-3+kk])/2

                # Modify the value in row
                neighborValues = repairImgDeblend[m[ii]-4:m[ii]+5, n[ii]]
                temp = neighborValues[neighborValues != 0]
                stdTemp = np.std(temp)
                meanTemp = np.mean(temp)

                for kk in range(9):
                    testValue = repairImgDeblend[m[ii]-4+kk, n[ii]]
                    if (testValue != 0):
                        if (testValue >= meanTemp + 2*stdTemp) or \
                           (testValue <= meanTemp - 2*stdTemp):
                            repairImgDeblend[m[ii]-4+kk, n[ii]] = \
                                (repairImgDeblend[m[ii]-5+kk, n[ii]] +
                                 repairImgDeblend[m[ii]-3+kk, n[ii]])/2

        return repairImgDeblend

    def _funcMag(self, magRatio, imgMainDonut, imgOverlapBinary, imgFit,
                 imgRef, xyShiftNeighbor):
        """Use the least square method to decide the magnitude ratio of
        neighboring star.

        Parameters
        ----------
        magRatio : float
            Magnitude ratio between the main star and neighboring star.
        imgMainDonut : numpy.ndarray[float]
            Image of the main star.
        imgOverlapBinary : numpy.ndarray[int]
            Binary image of overlap between the main star and neighboring star.
        imgFit : numpy.ndarray[float]
            Fitted image of neighboring star by the shifting of image of main
            star.
        imgRef : numpy.ndarray[float]
            Reference image for the fitting.
        xyShiftNeighbor : tuple or list
            Fitted image of neighboring star by the shifting of image of main
            star.

        Returns
        -------
        float
            The difference between synthesized image and blended image.
        """

        # Synthesize the image
        imgNew = imgMainDonut - magRatio*imgFit*imgOverlapBinary
        imgNew = imgNew + magRatio*shift(
            imgNew, [int(xyShiftNeighbor[1]), int(xyShiftNeighbor[0])])

        # Take the least square difference
        delta = np.sum((imgNew-imgRef)**2)

        return delta

    def _funcResidue(self, posShift, imgBinary, resImgBinary):
        """Use the least square method to decide the position of neighboring
        star.

        Parameters
        ----------
        posShift : tuple or list
            (x, y) shift from the main star position to neighboring star.
        imgBinary : numpy.ndarray[int]
            Binary image of the main star.
        resImgBinary : numpy.ndarray[int]
            Binary image of residue of neighboring star.

        Returns
        -------
        float
            The difference between fitted binary image and residue image.
        """

        # Shift the image
        fitImgBinary = shift(imgBinary, [int(posShift[1]), int(posShift[0])])

        # Handle the numerical error of shift. Regenerate a binary image.
        fitImgBinary[fitImgBinary > 0.5] = 1
        fitImgBinary[fitImgBinary < 0.5] = 0

        # Take the least square difference
        delta = np.sum((fitImgBinary-resImgBinary)**2)

        return delta

    def convolveExposureWithImage(self, exposure, kernelImage):

        '''Convolve image and variance planes in an exposure with an image using FFT
            Does not convolve mask. Returns new exposure'''

        newExposure = exposure.clone()

        image = self.convolveImageWithImage(newExposure.getImage(),
                                            kernelImage)
        variance = self.convolveImageWithImage(newExposure.getVariance(),
                                               kernelImage)

        newExposure.image = image
        newExposure.variance = variance
        return newExposure

    def convolveImageWithImage(self, image, kernelImage, conv=True, fft=True):

        '''Convolve/correlate an image with a kernel
            Option to use an FFT or direct (slow)
            Returns an image'''
        if conv:
            array = fftconvolve(image.getArray(), kernelImage.getArray(), mode='same')
        else:
            if fft:
                array = np.roll(ifft2(fft2(kernelImage.getArray()).conj()*fft2(image.getArray())).real,
                            (image.getArray().shape[0] - 1)//2, axis=(0, 1))
            else:
                array = correlate2d(image.getArray(), kernelImage.getArray(), mode='same')
        newImage = ImageF(array.shape[1], array.shape[0])
        newImage.array[:] = array
        return newImage


if __name__ == "__main__":
    pass
