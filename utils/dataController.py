import torch
import string
import codecs
import os

all_characters = string.printable
#converts each char in a word into a number and returns a tensor for that
def char_tensor(word):
    tensor = torch.zeros(len(word)).long()
    for c in range(len(word)):
        try:
            tensor[c] = all_characters.index(word[c])
        except:
            continue
    return tensor

def convertWordsToCharTensor(wordList, maxWordLen):
    tokens = len(wordList)
    wordIdTensor = torch.LongTensor(tokens, maxWordLen)     #Define a tensor to store all the ids in this wordList
    wordMaskTensor = torch.LongTensor(tokens, maxWordLen)
    token = 0												#define a token id
    for word in wordList:									#For every word in the list
        # Insert the tensor value
        padStr = ""
        seqLen = len(word)
        wordMask = torch.LongTensor([1 for x in range(seqLen)])
        if seqLen < maxWordLen:
            padLen = maxWordLen - seqLen
            padStr = "".join(["_" for x in range(padLen)])
            wordMask = torch.cat((wordMask, torch.LongTensor([0 for x in range(padLen)])))
        word += padStr
        # wordIdTensor[token] = self.dictionary.word2idx[word]
        wordIdTensor[token] = char_tensor(word)
        wordMaskTensor[token] = wordMask
        token += 1
    return wordIdTensor, wordMaskTensor  #Return tensors for the word and it's mask

def read_file_lines(filename):
    linesInFile = codecs.open(filename).readlines()
    return linesInFile

#This is a simple dictionary that you can use to get an index for words and get the words from an index, should be fairly straight forward
class WordDictionary(object):
    def __init__(self):
        self.word2idx = {}									#This is the dict that stores the indexes of the words, keyed by the words themselves
        self.idx2word = []									#This is the reverse that gets the word based on the index, the index is the list index

    #This function adds a word to the dictionary
    #word - the word that is to be added
    #returns a numerical index for the word
    def add_word(self, word):
        if word not in self.word2idx.keys():						#If we dont already have an index for the word, then we need to add it
            self.idx2word.append(word)						#Add the word to the list of words
            self.word2idx[word] = len(self.idx2word) - 1	#The index of that word then becomes the length of the word list -1 because it is now at the end
        return self.word2idx[word]							#Then we look up the word, and then return its index

    #This function gets the vocabulary size
    def __len__(self):
        return len(self.idx2word)							#Return the number of indexes that we have have, which is the vocabulary size

    def outputKey(self, path):
        f = open(path, "w")
        for word, idx in self.word2idx.items():
            f.write(str(word) + "->" + str(idx) + "\n")
        f.close()


#This is a simple class that stores all of our training, test, and validation data
class Corpus(object):
    #This is the constructor, it takes a folder path to the train, test, and validation data
    def __init__(self, filePrefix):
        self.dictionary = WordDictionary()

        path = filePrefix

        self.dictify(path + '.train')	#get all of the training examples
        self.dictify(path + '.valid')		#All of the validation examples
        self.dictify(path + '.test')    #and all of the testing examples

    #This function takes in a file path, reads it and then tokenizes the contents of each line in that file. The return value is a tensor (vector) that contains all the ids for the tokens in the file
    def dictify(self, path):
        # Add words to the dictionary
        with open(path, 'r') as f:										#Open the file
            for line in f.readlines():												#for every line in the file
                word = line.split()
                if len(word) > 0:
                    self.dictionary.add_word(word[0])

    def getWordIdx(self, word):
        idx = -1
        if word in self.dictionary.word2idx.keys():
            idx = self.dictionary.word2idx[word]
        return idx

    def idxToWord(self, idx):
        word = ""
        if idx < len(self.dictionary.idx2word):
            word = self.dictionary.idx2word[idx]
        return word
# labelCorpus = Corpus("twypo.label")
# print(len(labelCorpus.dictionary))