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

import sys

sys.path.append(".")
sys.path.append("../datasets")
sys.path.append("../utils")

from utils.helpers import *
from model import CharCNN
from utils.dataController import Corpus, read_file_lines, convertWordsToCharTensor


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
def random_training_set(batch_size, fileLinePtr, evaluation=False):
    numLines = len(fileLinePtr) - 1
    wordList = []
    labelList = []
    maxWordLen = 0
    target = torch.LongTensor(batch_size)
    for bi in range(batch_size):
        randomLine = fileLinePtr[rndm.randint(0, numLines)]
        words = randomLine.split()
        if len(words) < 2:
            # skip when we see an empty input or target
            bi -= 1
            continue
        if len(words[0]) > maxWordLen:
            maxWordLen = len(words[0])
        wordList.append(words[0])
        target[bi] = labelCorpus.getWordIdx(words[1])

    inp = torch.LongTensor(batch_size, maxWordLen)
    inpMask = torch.LongTensor(batch_size, maxWordLen)

    inp, inpMask = convertWordsToCharTensor(wordList, maxWordLen)
    inp = Variable(inp)
    inpMask = Variable(inpMask,volatile=evaluation)
    target = Variable(target,volatile=evaluation)
    if args.cuda:
        inp = inp.cuda()
        inpMask = inpMask.cuda()
        target = target.cuda()

    return inp, inpMask, target, wordList

def train(inp, inpMask, target, inputWordList):
    # hidden = decoder.init_hidden(args.batch_size)
    # if args.cuda:
    #     hidden = hidden.cuda()
    decoder.zero_grad()
    decoder.train()

    output = decoder(inp, inpMask)
    loss = criterion(output.view(args.batch_size, -1),target)

    # for c in range(args.chunk_len):
    #     output, hidden = decoder(inp[:,c], hidden)
    #     loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0]/args.batch_size

def save():
    save_filename = os.path.splitext(os.path.basename(args.fileName))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def evaluate(batch_size, fileLinePtr):
    decoder.eval()																						# Turn on evaluation mode which disables dropout.
    input, inputMask, target, _ = random_training_set(batch_size, fileLinePtr, True)
    output = decoder(input, inputMask)
    # Get the final output vector from the model (the typo suggestion word predicted)
    loss = criterion(output.view(args.batch_size, -1),target) # Get the loss of the predicitons
    return loss.data[0] / batch_size

def test(batch_size, fileLinePtr):
    decoder.eval()
    input, inputMask, target, input_word_list = random_training_set(batch_size, fileLinePtr, True)
    outputs = decoder(input, inputMask)
    temperature = 0.8
    for i in range(len(outputs)):
        # Sample from the network as a multinomial distribution
        output_dist = outputs[i].data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_word = labelCorpus.idxToWord(top_i)
        target_word = labelCorpus.idxToWord(target[i].data[0])
        print("Input:{}, Predicted:{} , Target:{}".format(input_word_list[i],predicted_word, target_word))

#number of input char types
char_vocab = len(string.printable)

# number of output classes = vocab size
numOutputClass = len(labelCorpus.dictionary)

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
            val_loss = evaluate(args.batch_size,linesInValid)  # test the model on validation data to check performance
            print('-' * 89)
            print("| end of epoch {%3d} %d%% | %s | valid loss {:%5.2f} | "%(epoch,
                epoch / args.n_epochs * 100,
                time_since(start),
                val_loss))  # Print some log statement
            print('-' * 89)
        #     print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()
    print("Testing...")
    test(len(linesInTest), linesInTest)

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
