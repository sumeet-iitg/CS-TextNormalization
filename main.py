# -*- coding: utf-8 -*-

from DataManagement.languageUtils import language, languageIdentifier, englishLanguage,hindiLanguage, polyglot_SpellChecker
from DataManagement.dataloader import monolingualCorpus
from DataManagement.filters import tweetFilterCollection
import codecs

def main():
    # srcFilePath = "sourceFile.txt"
    #
    # # load and normalize corpus
    # tweetFilter = tweetFilterCollection()
    # corpus = monolingualCorpus(srcFilePath, [tweetFilter],[englishLanguage])
    # corpus.applyFilters()
    inpFile = "sourceFile_filtered_lang.txt"
    outFileName = "sourceFile_lang_spellCheck.txt"
    with codecs.open(inpFile, 'r', 'utf-8') as fr:
        with codecs.open(outFileName, 'w', 'utf-8') as fw:
            print("Created outfile:" + outFileName)
            for line in fr.readlines():
                spellCorrected = polyglot_SpellChecker(line)
                fw.write(spellCorrected)

if __name__== "__main__":
    main()