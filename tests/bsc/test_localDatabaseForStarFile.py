import os
import shutil
import numpy as np
import unittest

from lsst.ts.wep.bsc.LocalDatabaseForStarFile import LocalDatabaseForStarFile
from lsst.ts.wep.Utility import getModulePath, FilterType


class TestLocalDatabaseForStarFile(unittest.TestCase):
    """Test the local database for star file class."""

    def setUp(self):

        self.modulePath = getModulePath()

        self.dataDir = os.path.join(self.modulePath, "tests", "tmp")
        self._makeDir(self.dataDir)

        self.filterType = FilterType.G
        self.db = LocalDatabaseForStarFile()

        dbAdress = os.path.join(self.modulePath, "tests", "testData",
                                "bsc.db3")
        self.db.connect(dbAdress)

    def _makeDir(self, directory):

        if (not os.path.exists(directory)):
            os.makedirs(directory)

    def tearDown(self):

        self.db.deleteTable(self.filterType)
        self.db.disconnect()

        shutil.rmtree(self.dataDir)

    def testTableIsInDb(self):

        self.assertFalse(self.db._tableIsInDb("StarTableG"))
        self.assertTrue(self.db._tableIsInDb("BrightStarCatalogU"))

    def testCreateTable(self):

        self._createTable()
        self.assertTrue(self.db._tableIsInDb("StarTableG"))

    def _createTable(self):
        self.db.createTable(self.filterType)

    def testCreateTableIfTableExist(self):

        self._createTable()
        self.assertRaises(ValueError, self.db.createTable, FilterType.G)

    def testInsertDataByFile(self):

        self._createTable()
        idAll = self.db.getAllId(self.filterType)
        self.assertEqual(len(idAll), 0)

        skyFilePath = os.path.join(self.modulePath, "tests", "testData",
                                   "skyComCamInfo.txt")
        idAll = self._insertDataToDbAndGetAllId(skyFilePath)

        self.assertEqual(len(idAll), 4)

    def _insertDataToDbAndGetAllId(self, skyFilePath):

        self.db.insertDataByFile(skyFilePath, self.filterType)
        idAll = self.db.getAllId(self.filterType)

        return idAll

    def testInsertDataByFileWithSingleStar(self):

        self._createTable()

        starData = [[1, 359.933039, -0.040709, 15.000000]]
        skyFilePath = self._writeStarFile("sglStar.txt", starData=starData)
        idAll = self._insertDataToDbAndGetAllId(skyFilePath)

        self.assertEqual(len(idAll), 1)

    def _writeStarFile(self, fileName, starData=[]):

        header = "Id     Ra      Decl        Mag"
        delimiter = "    "
        filePath = os.path.join(self.dataDir, fileName)
        np.savetxt(filePath, starData, delimiter=delimiter, header=header)

        return filePath

    def testInsertDataByFileWithoutStar(self):

        self._createTable()

        skyFilePath = self._writeStarFile("noStar.txt")
        with self.assertWarns(UserWarning):
            idAll = self._insertDataToDbAndGetAllId(skyFilePath)

        self.assertEqual(len(idAll), 0)

    def testDeleteTable(self):

        self._createTable()
        self.assertTrue(self.db._tableIsInDb("StarTableG"))

        self.db.deleteTable(self.filterType)
        self.assertFalse(self.db._tableIsInDb("StarTableG"))


if __name__ == "__main__":

    # Do the unit test
    unittest.main()
