# -*- coding: utf-8 -*-
import re, string, unicodedata
import os
import codecs
import emoji

# ekphasis toolkit from https://github.com/cbaziotis/ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector
from .constants import URL_REGEX, MENTION_REGEX, REPEAT_3_OR_MORE, REPEAT_STR_3_OR_MORE
from bs4 import BeautifulSoup

class filterCollection(object):
    def __init__(self):
        self.filters = []

    def doFiltering(self,inpFile,outFileName):
        return NotImplementedError

class dumbFilterCollection(filterCollection):
    def __init__(self):
        super().__init__()
        self.filters = [self.replaceUrl, self.replaceMention,
                        self.correctRepeatChars, self.correctRepeatStr, self.strip_unicode_punctuations]

    # some custom filters I was trying out.
    def stripHtml(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def replaceUrl(self, text):
        text = re.sub(URL_REGEX, "<URL>", text)
        return text

    def replaceMention(self, text):
        text = re.sub(MENTION_REGEX, "<USR>", text)
        return text

    def correctRepeatChars(self, text):
        text = re.sub(REPEAT_3_OR_MORE, r"\1", text)
        return text

    def strip_unicode_punctuations(self, text):
        clean_text = []
        for word in text.split():
          clean_text.append("".join(char for char in word if not unicodedata.category(char).startswith('P')))


    def correctRepeatStr(self, text):
        text = re.sub(REPEAT_STR_3_OR_MORE, r"\1", text)
        return text

    def doFiltering(self, inpPtr, outPtr):
        for line in inpPtr.readlines():
            for filter in self.filters:
                line = filter(line)
            outPtr.write(line)


class tweetFilterCollection(dumbFilterCollection):
    def __init__(self):
        super().__init__()
        self.ekphraPreProcessor = ekphraProc([emoticons])
        self.filters = [self.ekphrasize] #, self.replaceUrl, self.replaceMention]

    def ekphrasize(self,text):
        text = " ".join(self.ekphraPreProcessor.text_processor.pre_process_doc(text))
        return text

# wrapper around the ekphrasis preprocessor
class ekphraProc(object):
    def __init__(self, dictList):
        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis',},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter="twitter",

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector="twitter",

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)

            spell_correction = True, # do spell correction using word statistics
            spell_correct_elong=False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=False).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=dictList)
