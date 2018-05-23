class language:
    def __init__(self, languageCode):
        # dictionary of words in this language
        self.code = languageCode
        self.word2Idx = {}
        self.idx2Word = {}
        self.stopWordIds = []
        pass

    def addWordToDict(self, word, isStopWord = False):
        if not word in self.word2Idx.keys():
            lastDictPosn = len(self.word2Idx.keys()) - 1
            self.word2Idx[word] = lastDictPosn+1
            self.idx2Word[lastDictPosn+1] = word
            if isStopWord:
                self.stopWordIds.append(lastDictPosn+1)

class languageIdentifier:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def detectLanguageInSentence(self, sentence):
        pass

    def detectLanguageInWord(self, word):
        pass