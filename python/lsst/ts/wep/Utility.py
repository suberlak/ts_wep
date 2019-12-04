import os
import subprocess
import re
from scipy.ndimage.measurements import center_of_mass
from enum import Enum

from lsst.utils import getPackageDir


class FilterType(Enum):
    U = 1
    G = 2
    R = 3
    I = 4
    Z = 5
    Y = 6
    REF = 7

    @staticmethod
    def fromString(arg):
        """Instantiate the FilterType based on a string that maps
        one-to-one with the filters.

        Parameters
        ----------
        arg : str
            The string corresponding to the filter.

        Returns
        -------
        FilterType
            The FilterType corresponding to the string.
        """
        char = arg.strip().lower()
        string2Filter = {
            "u": FilterType.U,
            "g": FilterType.G,
            "r": FilterType.R,
            "i": FilterType.I,
            "z": FilterType.Z,
            "y": FilterType.Y,
            "ref": FilterType.REF
        }
        return string2Filter[char]

    def toString(self):
        """Represent the FilterType as a string.

        Returns
        -------
        str
            representation of self.
        """
        return str(self)

    def __str__(self):
        """Represent the FilterType as a string.

        Returns
        -------
        str
            representation of self.
        """
        filter2String = {
            FilterType.U: "u",
            FilterType.G: "g",
            FilterType.R: "r",
            FilterType.I: "i",
            FilterType.Z: "z",
            FilterType.Y: "y",
            FilterType.REF: "ref",
        }
        return filter2String[self]


class CamType(Enum):
    LsstCam = 1
    LsstFamCam = 2
    ComCam = 3
    PhoSim = 4


class BscDbType(Enum):
    LocalDb = 1
    LocalDbForStarFile = 2


class DefocalType(Enum):
    Intra = 1
    Extra = 2


class ImageType(Enum):
    Amp = 1
    Eimg = 2


class CentroidFindType(Enum):
    RandomWalk = 1
    Otsu = 2


def getModulePath():
    """Get the path of module.

    Returns
    -------
    str
        Directory path of module.
    """

    return getPackageDir("ts_wep")


def getConfigDir():
    """Get the directory of configuration files.

    Returns
    -------
    str
        Directory of configuration files.
    """

    return os.path.join(getModulePath(), "policy")


def getObsLsstCmdTaskConfigDir():
    """Get the obs_lsst command line task configuration directory.

    Returns
    -------
    str
        obs_lsst command line task configuration directory.
    """

    return os.path.join(getPackageDir("obs_lsst"), "config")


def runProgram(command, binDir=None, argstring=None):
    """Run the program w/o arguments.

    Parameters
    ----------
    command : str
        Command of application.
    binDir : str, optional
        Directory of binary application. (the default is None.)
    argstring : str, optional
        Arguments of program. (the default is None.)

    Raises
    ------
    RuntimeError
        Error running of command.
    """

    # Directory of binary application
    if (binDir is not None):
        command = os.path.join(binDir, command)

    # Arguments for the program
    if (argstring is not None):
        command += (" " + argstring)

    # Call the program w/o arguments
    if (subprocess.call(command, shell=True) != 0):
        raise RuntimeError("Error running: %s" % command)


def searchDonutPos(img):
    """Search the position of donut on image.

    Parameters
    ----------
    img : numpy.ndarray
         Donut image.

    Returns
    -------
    float
        X position of donut center in pixel.
    float
        Y position of donut center in pixel.
    """

    # Search the donut position by the center of mass
    # Need to update this method to the more robust one such as the convolution
    realcy, realcx = center_of_mass(img)

    return realcx, realcy


def writeFile(filePath, content):
    """Write the content to file.

    Parameters
    ----------
    filePath : str
        File path.
    content : str
        File content.
    """

    with open(filePath, "w") as file:
        file.write(content)


def readPhoSimSettingData(folderPath, fileName, atype):
    """Read the PhoSim setting data (segmentation or focal plane layout).

    Parameters
    ----------
    folderPath : str
        Path to folder.
    fileName : str
        File name ("segmentation.txt", "focalplanelayout.txt").
    atype : str
        Type of data to read ("readOutDim", "darkCurrent", "fieldCenter",
        "eulerRot").

    Returns
    -------
    dict
        Needed CCD data.

    Raises
    ------
    ValueError
        File can not be read.
    ValueError
        Type is not correct.
    """

    # Check the file name
    if fileName not in ("segmentation.txt", "focalplanelayout.txt"):
        raise ValueError("'%s' can not be read." % fileName)

    # Check the type
    if atype not in ("readOutDim", "darkCurrent", "fieldCenter", "eulerRot"):
        raise ValueError("'%s' can not be read." % atype)

    # Get the file path
    pathToFile = os.path.join(folderPath, fileName)

    # Amplifier list (only list the scientific ccd here)
    ampList = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07",
               "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17"]

    # Open the file to read
    ccdData = {}
    fid = open(pathToFile)
    for line in fid:
        line = line.strip()

        # Get each element
        lineElement = line.split()

        data = []
        # Analyze the sensor name to find the amplifier
        if (fileName == "segmentation.txt"):

            sensorNameStr = lineElement[0].split("_")
            if (len(sensorNameStr) == 3):
                if sensorNameStr[2] in ampList:
                    # Get the segmentation in txt file
                    if (atype == "readOutDim"):
                        # parallel prescan, serial overscan, serial prescan,
                        # parallel overscan (pixel)
                        data = lineElement[15:19]
                    elif (atype == "darkCurrent"):
                        data = lineElement[13:15]

        elif (fileName == "focalplanelayout.txt"):

            # Analyze the sensor name to make sure this line of data is
            # needed
            sensorNameStr = lineElement[0].split("_")
            if (len(sensorNameStr) == 2 or len(sensorNameStr) == 3):
                if (atype == "fieldCenter"):
                    # Collect the field center:
                    # x position (microns), y position (microns), pixel
                    # size (microns) number of x pixels, number of y pixels
                    data = lineElement[1:6]
                elif (atype == "eulerRot"):
                    # Collect the euler Rotation (degrees)
                    data = lineElement[10:13]

        # Collect the data
        if (data):
            ccdData.update({lineElement[0]: data})

    # Close the file
    fid.close()

    return ccdData


def mapFilterRefToG(filterType):
    """Map the reference filter to the G filter.

    Parameters
    ----------
    filterType : enum 'FilterType'
        Filter type.

    Returns
    -------
    enum 'FilterType'
        Mapped filter type.
    """

    if (filterType == FilterType.REF):
        return FilterType.G
    else:
        return filterType


def getBscDbType(bscDbType):
    """Get the bright star catalog (BSC) database type.

    Parameters
    ----------
    bscDbType : str
        BSC database type to use (localDb or file).

    Returns
    -------
    enum 'BscDbType'
        BSC database type.

    Raises
    ------
    ValueError
        The bscDb is not supported.
    """

    if (bscDbType == "localDb"):
        return BscDbType.LocalDb
    elif (bscDbType == "file"):
        return BscDbType.LocalDbForStarFile
    else:
        raise ValueError("The bscDb (%s) is not supported." % bscDbType)


def getImageType(imageType):
    """Get the image type.

    Parameters
    ----------
    imageType : str
        Image type to use (amp or eimage).

    Returns
    -------
    enum 'ImageType'
        ImageType enum.

    Raises
    ------
    ValueError
        The image type is not supported.
    """

    if (imageType == "amp"):
        return ImageType.Amp
    elif (imageType == "eimage"):
        return ImageType.Eimg
    else:
        raise ValueError("The %s is not supported." % imageType)


def getCentroidFindType(centroidFindType):
    """Get the centroid find type.

    Parameters
    ----------
    centroidFindType : str
        Centroid find algorithm to use (randomWalk or otsu).

    Returns
    -------
    enum 'CentroidFindType'
        Centroid find type algorithm.

    Raises
    ------
    ValueError
        The centroid find type is not supported.
    """

    if (centroidFindType == "randomWalk"):
        return CentroidFindType.RandomWalk
    elif (centroidFindType == "otsu"):
        return CentroidFindType.Otsu
    else:
        raise ValueError("The %s is not supported." % centroidFindType)


if __name__ == "__main__":
    pass
