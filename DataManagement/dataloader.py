# -*- coding: utf-8 -*-

import os

class corpus(object):
    def __init__(self, srcFilePath, filterCollection, langObjects):
        self.srcPath = srcFilePath
        self.filterCollection = filterCollection
        self.langObjects = langObjects

    def applyFilters(self):
        # TODO: this code will not work for now if there are more than one elems in filterCOllection
        assert len(self.filterCollection) < 2

        head, inpFileName = os.path.split(self.srcPath)
        fileName, ext = inpFileName.split(".")
        outFile = fileName + "_filtered"
        outFileName = os.path.join(outFile, ext)

        for filter in self.filterCollection:
            filter.doFiltering(self.srcPath, outFileName)

        return outFileName

    def normalize(self):
        return NotImplementedError

class monolingualCorpus(corpus):
    def __init__(self, srcFilePath, filters, langObjects):
        super().__init__(srcFilePath, filters, langObjects)

class bilingualCorpus(corpus):
    def __init__(self, srcFilePath, filters, langObjects):
        super().__init__(srcFilePath, filters, langObjects)


