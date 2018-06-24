# -*- coding: utf-8 -*-
import re
from DataManagement.constants import ANNOT_REGEX, LITCM_CODE_TO_UNIV_CODE

class language(object):
    def __init__(self, languageCode, spellChecker=None):
        # dictionary of words in this language
        self.code = languageCode
        self.word2Idx = {}
        self.idx2Word = {}
        self.stopWordIds = []

    def addWordToDict(self, word, isStopWord = False):
        if not word in self.word2Idx.keys():
            lastDictPosn = len(self.word2Idx.keys()) - 1
            self.word2Idx[word] = lastDictPosn+1
            self.idx2Word[lastDictPosn+1] = word
            if isStopWord:
                self.stopWordIds.append(lastDictPosn+1)

    '''
    If the word is in lexicon returns true, else false
    '''
    def basicSpellChecker(self, text):
        return True if text in self.word2Idx.keys() else False


class languageIdentifier(object):
    def __init__(self, languageSet):
        self.langSet = languageSet

    def detectLanguageInSentence(self, sentence):
        return NotImplementedError

    def detectLanguageInWord(self, word):
        return NotImplementedError

class indicLangIdentifier(languageIdentifier):
    def __init__(self,languageSet):
        super().__init__(languageSet)

def polyglot_SpellChecker(languageAnnotated_Text):
    correctedWords = []
    for wordLangPair in languageAnnotated_Text.split():
        word, lang = wordLangPair.split('\\')
        # if this word is annotation tag or has  then don't spell check
        if not re.search(ANNOT_REGEX, word) and lang.lower() in LITCM_CODE_TO_UNIV_CODE.keys():
            langObject = languageMap[LITCM_CODE_TO_UNIV_CODE[lang.lower()]]
            if not langObject.basicSpellChecker(word):
                word = "<unk>"
        correctedWords.append(word + '\\' + lang)
    return " ".join(correctedWords)

def getEnglish():
    englishLanguage = language("en")
    lexicon = []
    stopWords = []
    # create language dictionaries
    for word in lexicon:
        englishLanguage.addWordToDict(word, word in stopWords)
    return englishLanguage

def getHindi():
    hindiLanguage = language("hi")
    lexicon = []
    stopWords = []
    for word in lexicon:
        hindiLanguage.addWordToDict(word, word in stopWords)
    return hindiLanguage

englishLanguage = getEnglish()
hindiLanguage = getHindi()

# after getting language codes use this map to get the language specific objects
# which will help in spell checking
languageMap = {"en":englishLanguage, "hi":hindiLanguage}