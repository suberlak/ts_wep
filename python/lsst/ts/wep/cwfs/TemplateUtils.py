import os
import numpy as np
from lsst.ts.wep.Utility import DefocalType, getConfigDir, \
    CamType, abbrevDectectorName
from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage


def createTemplateImage(defocalState, sensorName, iniFieldXY,
                        templateType, donutImgSize):

    """
    Create/grab donut template.

    Parameters
    ----------
    sensorName : str
        Abbreviated sensor name.
    """

    configDir = getConfigDir()

    if templateType == 'phosim':
        if defocalState == DefocalType.Extra:
            template_filename = os.path.join(configDir, 'deblend',
                                             'data',
                                             'extra_template-%s.txt' %
                                             abbrevDectectorName(sensorName))
        elif defocalState == DefocalType.Intra:
            template_filename = os.path.join(configDir, 'deblend',
                                             'data',
                                             'intra_template-%s.txt' %
                                             abbrevDectectorName(sensorName))
        template_array = np.genfromtxt(template_filename)
        template_array[template_array < 50] = 0.

    elif templateType == 'model':
        # Load Instrument parameters
        instDir = os.path.join(configDir, "cwfs", "instData")
        dimOfDonutOnSensor = donutImgSize
        inst = Instrument(instDir)
        inst.config(CamType.LsstCam, dimOfDonutOnSensor)

        # Create image for mask
        img = CompensableImage()
        img.defocalType = defocalState

        # define position of donut at center of current sensor
        boundaryT = 0
        maskScalingFactorLocal = 1
        img.fieldX, img.fieldY = iniFieldXY[0]
        img.makeMask(inst, "offAxis", boundaryT, maskScalingFactorLocal)

        template_array = img.cMask

    return template_array
