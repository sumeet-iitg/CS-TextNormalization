# -*- coding: utf-8 -*-

'''
As of now this file is only to execute the normalization process from a single place
'''

from DataManagement.languageUtils import language, languageIdentifier, englishLanguage,hindiLanguage, polyglot_SpellChecker
from DataManagement.dataloader import monolingualCorpus
from DataManagement.filters import tweetFilterCollection, dumbFilterCollection
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
    with codecs.open(inpFile, 'r', encoding='utf-8') as fr:
        with codecs.open(outFileName, 'w', encoding='utf-8') as fw:
            print("Created outfile:" + outFileName)
            for line in fr.readlines():
                spellCorrected = polyglot_SpellChecker(line)
                fw.write(spellCorrected)

'''
This applies some basic filters on tweets
'''
def normalize_ner_tweets(tweetsFile):
    dumbFilter = dumbFilterCollection()

    corpus = monolingualCorpus(tweetsFile, [dumbFilter], [englishLanguage])
    corpus.applyFilters()

if __name__== "__main__":
    # main()
    tweetsFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\calcs_train_tweets.tsv"
    normalize_ner_tweets(tweetsFile)