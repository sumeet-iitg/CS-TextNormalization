# -*- coding: utf-8 -*-
from DataManagement.languageUtils import language, languageIdentifier
import codecs
from DataManagement.filters import tweetFilter, blogFilter

class corpus(object):
    def __init__(self, srcFilePath):
        self.srcPath = srcFilePath

class biLingualCorpus(corpus):
    def __init__(self, srcFilePath, filters, langId1, langId2):
        super().__init__(srcFilePath)
        self.langId1 = langId1
        self.langId2 = langId2
        self.filters = filters

    def applyFilters(self):
        tmpInpFilePath = "/tmp/filterIn.txt"
        # copy contents from input file path to this temp file
        for filter in self.filters:
            filter(tmpInpFilePath)

        return tmpInpFilePath

class twitterBiLingual(biLingualCorpus):
    def __init__(self, srcFilePath, filters, langId1, langId2):
        super().__init__(srcFilePath, filters, langId1, langId2)

    # annotates each token in the word to be either Hashtags, Mentions, Url or Content
    def tokenizer(self, line):
        tokenizedLine = {}
        id = 0
        for word in line.split():
            # type can be content, hashtag, mention or url
            tokenizedLine[id] = {"type":'content', "val":word, "lang":"en"}
            id += 1
        return tokenizedLine
