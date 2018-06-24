# -*- coding: utf-8 -*-
import re, string, unicodedata
import os
import codecs
# ekphasis toolkit from https://github.com/cbaziotis/ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from DataManagement.constants import URL_REGEX, MENTION_REGEX

from bs4 import BeautifulSoup

class filterCollection(object):
    def __init__(self):
        self.filters = []

    def doFiltering(self,inpFile,outFileName):
        return NotImplementedError

class dumbFilterCollection(filterCollection):
    def __init__(self):
        super().__init__()
        self.filters = [self.stripHtml, self.replaceUrl, self.replaceMention]

    # some custom filters I was trying out.
    def stripHtml(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def replaceUrl(self, text):
        text = re.sub(URL_REGEX, text, "<url>")
        return text

    def replaceMention(self, text):
        text = re.sub(MENTION_REGEX, text, "<user>")
        return text

    def doFiltering(self, inpFile, outFileName):
        print("Inside doFiltering")
        with codecs.open(inpFile, 'r', 'utf-8') as fr:
            with codecs.open(outFileName, 'w', 'utf-8') as fw:
                print("Created outfile:" + outFileName)
                for line in fr.readlines():
                    for filter in self.filters:
                        line = filter(line)
                    fw.write(line + "\n")

class tweetFilterCollection(dumbFilterCollection):
    def __init__(self):
        super().__init__()
        self.filters = [self.ekphrasize] #, self.replaceUrl, self.replaceMention]

    def ekphrasize(self,text):
        ekphraPreProcessor = ekphraProc([emoticons])
        text = " ".join(ekphraPreProcessor.text_processor.pre_process_doc(text))
        return text


# wrapper around the ekphrasis preprocessor
class ekphraProc(object):
    def __init__(self, dictList):
        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter="twitter",

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector="twitter",

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=dictList)
