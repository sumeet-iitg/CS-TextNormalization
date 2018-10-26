# -*- coding: utf-8 -*-
"""
Usage:
    main.py

Options:
    -h --help                               show this screen.
    --source-file=<file>                    source file to cleanse
                                            [default:  'file.txt']
    --lang-set=<string>                     list of languages in the corpus
                                            [default: ['eng','hin']]
    --lang-file=<file>                      language annotated file
                                            [default:  'file.txt']
    --aggressiveness=<float>                aggressiveness of cleaning
                                            [default: 1.0]
"""

from DataManagement import indicLangIdentifier, polyglot_SpellChecker, indic_transliterator
from DataManagement import monolingualCorpus
from DataManagement import tweetFilterCollection, dumbFilterCollection
from cm_spellchecker import Spellchecker
import codecs
from docopt import docopt
import os


def main():
    inpFile = "sourceFile_filtered_lang.txt"
    outFileName = "sourceFile_lang_spellCheck.txt"
    with codecs.open(inpFile, 'r', encoding='utf-8') as fr:
        with codecs.open(outFileName, 'w', encoding='utf-8') as fw:
            print("Created outfile:" + outFileName)
            for line in fr.readlines():
                spellCorrected = polyglot_SpellChecker(line)
                fw.write(spellCorrected)

def normalize_hinglish_tweets(source_file, lang_list):
    '''
    :param source_file: file containing tweets
    :param lang_list: list of languages with which to condition the language identifier
    :return: text cleaned from #tags, RT, transliterated and spell-corrrected
    '''
    dumbFilter = dumbFilterCollection()
    hinglish_lid = indicLangIdentifier(lang_list)


    head, inpFileName = os.path.split(source_file)
    fileName, ext = inpFileName.split(".")
    outFile = fileName + "_filtered"
    outFile = os.path.join(head, outFile + "." + ext)

    with codecs.open(source_file, 'r', encoding='utf-8') as fr:
        with codecs.open(outFile, 'w', encoding='utf-8') as fw:
            for line in fr.readlines():
                # 1. Apply basic filtering
                line = dumbFilter.filterLine(line)
                # 2. Language Tag the line
                lid_tags = []
                for word in line.split():
                    lid_tags.append(hinglish_lid.detectLanguageInWord(word))
                # 3. Transliterate each word to their language specific script
                translit_words = []
                for word,lang in zip(line,lid_tags):
                    translit_words.append(indic_transliterator(word, "english","telugu"))
                fw.write(" ".join(translit_words))
    return outFile

def normalize_codemixed_text(source_file, lang_list):
    '''
    :param source_file: file containing tweets
    :param lang_list: list of languages with which to condition the language identifier
    :return: text cleaned from #tags, RT, transliterated and spell-corrrected
    '''
    dumbFilter = dumbFilterCollection()

    # loads a language identifier
    lid = indicLangIdentifier(lang_list)

    head, inpFileName = os.path.split(source_file)
    fileName, ext = inpFileName.split(".")
    outFile = fileName + "_filtered"
    outFile = os.path.join(head, outFile + "." + ext)
    spellChecker = Spellchecker()

    # if the lines within this file are already language annotated
    isLangTagged = True

    with codecs.open(source_file, 'r', encoding='utf-8') as fr:
        with codecs.open(outFile, 'w', encoding='utf-8') as fw:
            for line in fr.readlines():
                # 1. Apply basic filtering
                line = dumbFilter.filterLine(line)
                # 2. Language Tag the line
                lid_tags = []
                words = []
                lang_tagged_line = ""
                for token in line.split():
                    if isLangTagged:
                        word,lang = token.split('\\')
                        words.append(word)
                        lid_tags.append(token)
                        lang_tagged_line +=token
                    else:
                        words.append(token)
                        lang = lid.detectLanguageInWord(token)
                        lang_tagged_line += token + "\\" + lang
                        lid_tags.append(lang)

                spell_corrected_line = spellChecker.correctSentence(lang_tagged_line)
                # 3. Transliterate each word to their language specific script
                translit_words = []

                for word, lang in zip(spell_corrected_line.split(" "), lid_tags):
                    translit_words.append(indic_transliterator(word, "english", lang))

                fw.write(" ".join(translit_words))

    return outFile

if __name__== "__main__":
    # main()
    # eng-spa tweetsFile = "C:\\Users\\SumeetSingh\\Documents\\Code-Mixed\\ACL-CM-NER-2018-eng-spa\\calcs_train_tweets.tsv"
    args = docopt(__doc__)
    source_file = "../sourceFile.txt"
    lang_file = ""
    langList = ['english','hindi']
    # normalize_hinglish_tweets(source_file, langList)
    langList = ['english','telugu']
    normalize_codemixed_text(source_file, langList)