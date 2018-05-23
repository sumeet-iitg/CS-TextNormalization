from DataManagement.languageUtils import language, languageIdentifier
from DataManagement.dataloader import blogCorpus


# declare language pairs
englishWords = []
englishStopWords = []
hindiWords = []
hindiStopWords = []


englishLanguage = language("en")
hindiLanguage = language("hi_IN")

# create language dictionaries
for word in englishWords:
    englishLanguage.addWordToDict(word, word in englishStopWords)

for word in hindiWords:
    hindiLanguage.addWordToDict(word, word in hindiStopWords)

srcFilePath = "sourceFile.txt"
targetFilePath = "targetFile.txt"

# load and normalize corpus
corpus = blogCorpus(srcFilePath, targetFilePath, englishLanguage, hindiLanguage,\
                    languageIdentifier("en", "hi_IN"))

corpus.normalize()