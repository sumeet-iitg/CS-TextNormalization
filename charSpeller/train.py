#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import random as rndm
import string
import time

from tqdm import tqdm

from helpers import *
from model import CharCNN
from generate import *
from dataController import Corpus, read_file_lines,convertWordsToCharTensor


# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--fileName', type=str)
argparser.add_argument('--labelFile', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

# file, file_len = read_file(args.filename)
trainFile= args.fileName + ".train"
validFile = args.fileName + ".valid"
testFile = args.fileName + ".test"

labelTrainFile = args.labelFile + ".train"
labelValidFile = args.labelFile + ".valid"
labelTestFile = args.labelFile + ".test"

labelCorpus = Corpus(args.labelFile)

linesInTrain = read_file_lines(trainFile)
linesInValid = read_file_lines(validFile)
linesInTest = read_file_lines(testFile)


#chunk_len is a fixed size of the input with padding
def random_training_set(batch_size, fileLinePtr):
    numLines = len(fileLinePtr)
    wordList = []
    labelList = []
    maxWordLen = 0
    for bi in range(batch_size):
        randomLine = fileLinePtr[rndm.randint(0, numLines)]
        words = randomLine.split()
        if len(words[0]) > maxWordLen:
            maxWordLen = len(words[0])
        wordList.append(words[0])
        labelList.append(words[1])

    inp = torch.LongTensor(batch_size, maxWordLen)
    inpMask = torch.LongTensor(batch_size, maxWordLen)
    target = torch.LongTensor(batch_size)

    inp, inpMask = convertWordsToCharTensor(wordList)
    for word in labelList:
        target.append(labelCorpus.getWordIdx(word))

    inp = Variable(inp)
    inpMask = Variable(inpMask)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        inpMask = inpMask.cuda()
        target = target.cuda()

    return inp, inpMask, target

def train(inp, inpMask, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()

    output, hidden = decoder(inp, inpMask, hidden)
    loss = criterion(output.view(args.batch_size, -1),target)

    # for c in range(args.chunk_len):
    #     output, hidden = decoder(inp[:,c], hidden)
    #     loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0]

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

#number of input char types
char_vocab = len(string.printable)

# number of output classes = vocab size
numOutputClass = len(labelCorpus.dictionary.keys())

# Initialize models and start training

decoder = CharCNN(
    char_vocab,
    args.hidden_size,
    numOutputClass)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.batch_size,linesInTrain))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
        #     print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
