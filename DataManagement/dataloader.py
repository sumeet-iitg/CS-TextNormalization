# -*- coding: utf-8 -*-
from DataManagement.languageUtils import language, languageIdentifier
import codecs


class codeSwitchedCorpus:
    def __init__(self, filePath, tgtFilePath, lang1, lang2, langIdentifier):
        self.filePath = filePath
        self.tgtFilePath = tgtFilePath
        self.lang1 = lang1
        self.lang2 = lang2
        self.langIdentifier = langIdentifier

    '''
    Reads through the corpus, tokenizing it and annotating with language 
    '''
    def normalize(self):
        with codecs.open(self.filePath, 'r', encoding='utf-8') as rdr:
            with codecs.open(self.tgtFilePath, 'w', encoding='utf-8') as wtr:
                for line in rdr:
                    tokenizedLine = self.tokenizer(line)
                    wtr.write(self.normalizeLine(tokenizedLine))

    def tokenizer(self, line):
        tokenizedLine = {}
        return tokenizedLine

    '''
    Normalizes a sentenc annotated with language
    '''
    def normalizeLine(self, tokenizedLine):
        return NotImplementedError

class twitterCorpus(codeSwitchedCorpus):
    def __init__(self, srcFilePath, tgtFilePath, lang1, lang2, langIdentifier):
        super().__init__(srcFilePath, tgtFilePath, lang1, lang2, langIdentifier)

    # annotates each token in the word to be either Hashtags, Mentions, Url or Content
    def tokenizer(self, line):
        tokenizedLine = {}
        id = 0
        for word in line.split():
            # type can be content, hashtag, mention or url
            tokenizedLine[id] = {"type":'content', "val":word, "lang":"en"}
            id += 1
        return tokenizedLine

# decide if blog corpus should be built along the lines of twitter corpus
class blogCorpus(twitterCorpus):
    def __init__(self, srcFilePath, tgtFilePath, lang1, lang2, langIdentifier):
        super().__init__(srcFilePath, tgtFilePath, lang1, lang2, langIdentifier)

