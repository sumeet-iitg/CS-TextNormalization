import os
from glob import glob
import codecs
from collections import defaultdict


data_dir = "C:\\Users\\Sumeet Singh\\PycharmProjects\\TextNormalization\\CS-TextNormalization\\Equilid\\lang76.100k"


source_vocab_set = set()
# get all source
source_files = glob(data_dir + '/*'+ ".source*ids")
for source_file in source_files:
    print("Reading from file-path {}".format(source_file))
    with codecs.open(source_file, 'r', encoding='utf-8') as source_fp:
        source_line = source_fp.readline().strip()
        while source_line:
            source_vocab_set.update(id for id in source_line.split())
            source_line = source_fp.readline().strip()
print("length of source vocab = {}".format(len(source_vocab_set)))

source_vocab_file = data_dir + '/' + "source_vocab.txt"
with open(source_vocab_file, 'w') as src_fp:
    for id in source_vocab_set:
        src_fp.write(id + '\n')