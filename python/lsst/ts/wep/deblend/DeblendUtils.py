import os
import numpy as np
from scipy.signal import fftconvolve, correlate2d
from numpy.fft import fft2, ifft2
from lsst.afw.image import ImageF
from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
from lsst.ts.wep.Utility import getConfigDir, CamType
from lsst.ts.wep.ParamReader import ParamReader


class DeblendUtils():

    def __init__(self, settingFileName="default.yaml", **kwargs):

        self.configDir = getConfigDir()
        settingFilePath = os.path.join(self.configDir, settingFileName)
        self.settingFile = ParamReader(filePath=settingFilePath)
        self.templateType = self.settingFile.getSetting('templateType')
        super().__init__(**kwargs)

    def createTemplateImage(self, sensor_name, defocal_type):

        """
        Create/grab donut template.
        """

        if self.templateType == 'phosim':
            if defocal_type == 'extra':
                template_filename = os.path.join(self.configDir, 'deblend',
                                                 'data',
                                                 'extra_template-%s.txt' %
                                                 sensor_name)
            elif defocal_type == 'intra':
                template_filename = os.path.join(self.configDir, 'deblend',
                                                 'data',
                                                 'intra_template-%s.txt' %
                                                 sensor_name)
            template_array = np.genfromtxt(template_filename)
            template_array[template_array < 50] = 0.

        if self.templateType == 'model':
            # Load Instrument parameters
            instDir = os.path.join(self.configDir, "cwfs", "instData")
            dimOfDonutOnSensor = \
                self.settingFile.getSetting('donutImgSizeInPixel')
            inst = Instrument()
            inst.config(CamType.LsstCam, dimOfDonutOnSensor)

            # Create image for mask
            img = CompensableImage()

            # define position of donut
            boundaryT = 0
            maskScalingFactorLocal=1

        return template_array

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
