class language(object):
    def __init__(self, languageCode, spellChecker=None):
        # dictionary of words in this language
        self.code = languageCode
        self.word2Idx = {}
        self.idx2Word = {}
        self.stopWordIds = []
        self.spellChecker = spellChecker

    def addWordToDict(self, word, isStopWord = False):
        if not word in self.word2Idx.keys():
            lastDictPosn = len(self.word2Idx.keys()) - 1
            self.word2Idx[word] = lastDictPosn+1
            self.idx2Word[lastDictPosn+1] = word
            if isStopWord:
                self.stopWordIds.append(lastDictPosn+1)

class languageIdentifier(object):
    def __init__(self, languageSet):
        self.langSet = languageSet

    def detectLanguageInSentence(self, sentence):
        pass

    def detectLanguageInWord(self, word):
        pass

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