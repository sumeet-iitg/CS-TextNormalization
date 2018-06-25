# -*- coding: utf-8 -*-
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import codecs

with codecs.open(
        "C:\\Users\\Sumeet Singh\\PycharmProjects\\TextNormalization\\CS-TextNormalization\\datasets\\top1000-Hindi.txt",'r', encoding='utf-8') as fr:
    with codecs.open("C:\\Users\\Sumeet Singh\\PycharmProjects\\TextNormalization\\CS-TextNormalization\\datasets\\top1000-Hindi-translit.txt", 'w', encoding='utf-8') as fw:
        for line in fr.readlines():
            data = line.split()
            if len(data) > 0:
                romanized = transliterate(data[0], sanscript.DEVANAGARI, sanscript.ITRANS)
                fw.write(romanized.lower()+'\n')
