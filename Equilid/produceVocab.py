import os
from glob import glob
import codecs
from collections import defaultdict


data_dir = "C:\\Users\\Sumeet Singh\\PycharmProjects\\TextNormalization\\CS-TextNormalization\\Equilid\\lang76.100k"

src_vocab_file = os.path.join(data_dir, 'vocab.src')
tgt_vocab_file = os.path.join(data_dir, 'vocab.tgt')
src_vocab = defaultdict(lambda:-1)

data_type_ext = ".source.train"
src_train_file_set= set()
# get all source files
for source_file in glob(data_dir + '/*'+ data_type_ext + '*'):
    base_name = os.path.basename(source_file)
    src_train_file_set.add(base_name.split(".")[0])

# for each source filename make a pair of txt and ids file
src_train_file_pair = []
for src_file in src_train_file_set:
    src_train_file_pair.append((os.path.join(data_dir,src_file + data_type_ext + '.txt'), os.path.join(data_dir, src_file + data_type_ext + '.ids')))

for file_txt,file_id in src_train_file_pair:
    break_flag = 0
    with codecs.open(file_txt, 'r', encoding='utf-8') as file_txt_fp, open(file_id, 'r', encoding='utf-8') as file_id_fp:
        print("Analyzing {} and {}".format(file_txt,file_id))
        for text in zip(file_txt_fp, file_id_fp):
            if break_flag:
                print("Skipping rest of the file due to mis-match.")
                break_flag = 0
                break

            words,ids = text
            words  = words.split('\t')[0]
            char_list = list(words.strip("\r\n"))
            id_list = ids.strip("\r\n").split()
            if not len(char_list) == len(id_list):
                print("========Length Mismatch=====")
                print("Words:{}, Sentence:{} length:{} \n Ids:{} length:{}".format(words, char_list,len(char_list),id_list,len(id_list)))
                break_flag = 1
                continue
            for i in range(len(char_list)):
                if src_vocab.get(char_list[i]) == -1:
                    # the previously stored mapping for char should be the same
                    print("========Inconsistent Map========")
                    print("Words:{}, Sentence:{} length:{} \n Ids:{} length:{}".format(words, char_list, len(char_list),
                                                                                     id_list, len(id_list)))
                    print(src_vocab[char_list[i]], id_list[i])
                    break_flag = 1
                    break
                src_vocab[char_list[i]] = id_list[i]

# get length of total number of words in the vocab
with codecs.open(src_vocab_file, "r",encoding='utf-8') as vocab_fp:
    src_vocab_len = len(vocab_fp.readlines())

print(len(src_vocab), src_vocab_len)

with codecs.open(os.path.join(data_dir,'src_vocab.dict'),'w',encoding='utf-8') as dict_fp:
    for k,v in src_vocab.items():
        dict_fp.write(k+'\t'+v+'\n')