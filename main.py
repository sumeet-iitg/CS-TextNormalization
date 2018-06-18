from DataManagement.languageUtils import language, languageIdentifier, englishLanguage
from DataManagement.dataloader import monolingualCorpus
from DataManagement.filters import tweetFilterCollection


def main():
    srcFilePath = "sourceFile.txt"

    # load and normalize corpus
    corpus = monolingualCorpus(srcFilePath, englishLanguage, tweetFilterCollection,[englishLanguage])
    corpus.applyFilters()
    # corpus.normalize()

if __name__=="main":
    main()