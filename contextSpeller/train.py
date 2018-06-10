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
from model import CharCNN, DecoderRNN
from utils.dataController import SentenceCorpus, read_file_lines, convertWordsToCharTensor,SOS


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

labelCorpus = SentenceCorpus(args.labelFile)

linesInTrain = read_file_lines(trainFile)
linesInValid = read_file_lines(validFile)
linesInTest = read_file_lines(testFile)

#chunk_len is a fixed size of the input with padding

def get_minibatch(fileLinePtr, startIndex, batch_size, evaluation=False):
    target = torch.LongTensor(batch_size, 3)
    inpSentenceList = []
    maxSentLen = 0

    for bi in range(batch_size):
        randomLine = fileLinePtr[startIndex + bi]
        sentences = randomLine.split()
        if len(sentences) < 2:
            # skip when we see an empty input or target
            bi -= 1
            continue

        if len(sentences[0]) > maxSentLen:
            maxSentLen = len(sentences[0])
        inpSentenceList.append(sentences[0])
        targetWords = sentences[1].split('_')
        for i in range(len(targetWords)):
            targetClass = labelCorpus.getWordIdx(targetWords[i])
            if targetClass >= numOutputClass or targetClass < 0:
                print("Target class:{}, for word:{}".format(str(targetClass), str(targetWords[i])))
            target[bi][i] = targetClass

    inp = torch.LongTensor(batch_size, maxSentLen)
    inpMask = torch.LongTensor(batch_size, maxSentLen)

    # A sentence is formed with words linked by '_' followed by additional '_'
    inp, inpMask = convertWordsToCharTensor(inpSentenceList, maxSentLen)
    inp = Variable(inp)
    inpMask = Variable(inpMask,volatile=evaluation)

    target = Variable(target,volatile=evaluation)
    if args.cuda:
        inp = inp.cuda()
        inpMask = inpMask.cuda()
        target = target.cuda()

    return inp, inpMask, target, inpSentenceList


def random_training_set(batch_size, fileLinePtr, evaluation=False):
    numLines = len(fileLinePtr) - 1
    inpSentenceList = []
    outSentenceList = []
    labelList = []
    maxSentLen = 0
    target = torch.LongTensor(batch_size, 3)
    for bi in range(batch_size):
        randomLine = fileLinePtr[rndm.randint(0, numLines)]
        sentences = randomLine.split()
        if len(sentences) < 2:
            # skip when we see an empty input or target
            bi -= 1
            continue

        if len(sentences[0]) > maxSentLen:
            maxSentLen = len(sentences[0])
        inpSentenceList.append(sentences[0])
        targetWords = sentences[1].split('_')
        for i in range(len(targetWords)):
            targetClass = labelCorpus.getWordIdx(targetWords[i])
            if targetClass >= numOutputClass or targetClass <0:
                print("Target class:{}, for word:{}".format(str(targetClass), str(targetWords[i])))
            target[bi][i] = targetClass

    inp = torch.LongTensor(batch_size, maxSentLen)
    inpMask = torch.LongTensor(batch_size, maxSentLen)

    # A sentence is formed with words linked by '_' followed by additional '_'
    inp, inpMask = convertWordsToCharTensor(inpSentenceList, maxSentLen)
    inp = Variable(inp)
    inpMask = Variable(inpMask,volatile=evaluation)

    target = Variable(target,volatile=evaluation)
    if args.cuda:
        inp = inp.cuda()
        inpMask = inpMask.cuda()
        target = target.cuda()

    return inp, inpMask, target, inpSentenceList


teacherForcingRatio = 0.5
def train(inp, inpMask, target, inpSentenceList):
    batch_size = target.size(0)
    inp_len = target.size(1)
    encoder.zero_grad()
    encoder.train()
    decoder.zero_grad()
    decoder.train()
    encodedOutput = encoder(inp, inpMask)
    teacherForcing = True if rndm.random() < teacherForcingRatio else False
    decoder_input = Variable(torch.LongTensor([labelCorpus.getWordIdx(SOS) for x in range(batch_size)]))
    decoder_hidden = encodedOutput[0].unsqueeze(0)
    loss = 0

    if teacherForcing:
        for di in range(3):# target sentence length to predict is always 3
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            # maxClassProb, predictedClasses = torch.max(decoder_output, 1)
            # print("predictedClass, target:" + str(predictedClasses.size()) + "," + str(target[:,di].size()))
            loss += criterion(decoder_output, target[:,di])
            decoder_input = target[:,di]  # Teacher forcing
    else:
        for di in range(3):# target sentence length to predict is always 3
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # maxClassProb, predictedClasses = torch.max(decoder_output, 1)
            # print("predictedClass, target:" + str(predictedClasses.size()) + "," + str(target[:,di].size()))
            loss += criterion(decoder_output, target[:,di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]/(args.batch_size*3)

'''
Runs a sequence decoder on a batch of initial inputs and returns the total loss.
If teacher forcing is enabled then uses the ground truth from target 
'''
def runDecoder(decoderObj, decoderInput, decoderHidden, targets, doTeacherForcing=False):
    seqLen = targets.size(1)
    totalLoss = 0
    decoder_outputs = []
    for di in range(seqLen):
        decoder_output, decoder_hidden = decoderObj(
            decoderInput, decoderHidden)

        decoder_outputs.append(decoder_output)
        # print("predictedClass, target:" + str(predictedClasses.size()) + "," + str(target[:,di].size()))
        totalLoss += criterion(decoder_output, targets[:, di])

        if doTeacherForcing:
            decoderInput = targets[:, di]  # Teacher forcing
        else:
            topv, topi = decoder_output.topk(1)
            decoderInput = topi.squeeze().detach()  # detach from history as input

    concatenated_output = torch.stack(decoder_outputs, dim=1)
    return totalLoss,concatenated_output

def save():
    save_filename = os.path.splitext(os.path.basename(args.fileName))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

def evaluate(fileLinePtr, printPred = False):
    encoder.eval()
    decoder.eval() # Turn on evaluation mode
    loss = 0
    numLines = len(fileLinePtr)
    totalLoss = 0

    for j in range(0, numLines, args.batch_size):
        batch_size = args.batch_size
        if j + args.batch_size >= numLines:
            batch_size = numLines - j
        input, inputMask, target, input_word_list = get_minibatch(fileLinePtr, j, batch_size, True)
        encoderOutput = encoder(input, inputMask)
        decoder_input = Variable(torch.LongTensor([labelCorpus.getWordIdx(SOS) for x in range(batch_size)]), volatile = True)
        decoder_hidden = encoderOutput[0].unsqueeze(0)
        currLoss, outputs = runDecoder(decoder, decoder_input,decoder_hidden, target)# no teacher forcing during evaluation
        totalLoss += currLoss
        temperature = 0.8
        if printPred:
            # decoder_output & target are of dim batch_size x seq_len
            for row in range(outputs.size(0)):
                # Sample from the network as a multinomial distribution
                predicted_word = ""
                target_word = ""
                for i in range(outputs.size(1)):
                    output_dist = outputs[row][i].data.view(-1).div(temperature).exp()
                    top_i = torch.multinomial(output_dist, 1)[0]
                    predicted_word += labelCorpus.idxToWord(top_i) + "_"
                    target_word += labelCorpus.idxToWord(target[row][i].data[0]) + "_"
                print("Input:{}, Predicted:{} , Target:{}".format(input_word_list[row], predicted_word, target_word))
    return totalLoss / numLines

#number of input char types
char_vocab = len(string.printable)

# number of output classes = vocab size
numOutputClass = len(labelCorpus.dictionary)
print("Number of Classes:" + str(numOutputClass))

# Initialize models and start training

encoder = CharCNN(
    char_vocab,
    args.hidden_size)

decoder = DecoderRNN(args.hidden_size, numOutputClass)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    criterion.cuda()
    encoder.cuda()
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    numMiniBatches = len(linesInTrain)/args.batch_size

    for epoch in tqdm(range(1, args.n_epochs + 1)):
        minibatchesSinceLastPrint = 0
        for j in range(0, len(linesInTrain), args.batch_size):
            batch_size = args.batch_size
            if j + args.batch_size >= len(linesInTrain):
                batch_size = len(linesInTrain) - j
            loss = train(*get_minibatch(linesInTrain, j, batch_size))
            loss_avg += loss
            minibatchesSinceLastPrint += 1
            if minibatchesSinceLastPrint >= (args.print_every) :
                print("current batch  %d-%d| %s | train loss {:%5.2f} | " % (j + 1,
                                                                            j + batch_size + 1,
                                                                            time_since(start),
                                                                            loss))  # Print some log statement
                # val_loss = evaluate(linesInValid, True)
                minibatchesSinceLastPrint = 0
            # Print some log statement
        val_loss = evaluate(linesInValid, False)  # test the model on entire validation data
        print('-' * 89)
        print("| end of epoch {%3d} %d%% | %s | valid loss {:%5.2f} | "%(epoch,
                epoch / args.n_epochs * 100,
                time_since(start),
                val_loss))  # Print some log statement
        print('-' * 89)
        #     print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()

    # print("Testing...")
    # test(len(linesInTest), linesInTest)

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
