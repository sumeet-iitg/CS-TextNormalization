# inspired from https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, num_input_chars, hidden_size, model = "gru", n_layers = 1):
        super(CharCNN, self).__init__()
        self.model = model.lower()
        self.input_size = num_input_chars
        self.hidden_size = hidden_size

        self.lookup = nn.Embedding(num_input_chars, hidden_size)
        self.conv1d = nn.Conv1d(hidden_size,hidden_size,2) #inChannel, outChannel, kH, kWid
        self.pool1d = nn.AvgPool1d(2) # kernel width over which to pool

        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)

    def forward(self, inputs, inputMasks):

        batch_size = inputs.size(0)
        inp_width = inputs.size(1)

        # print("input:" + str(inputs.size()))
        # Input has dim batch_size x max_word_len
        # Embedding expects 2-d input and replaces every element
        # with a vector. Thus the order of the dimensions of the input has no importance.
        # this is (batch_size x seq_len x embedding_size)
        char_embeddings = self.lookup(inputs)

        # Turn (batch_size x word_len x embedding_size) into (batch_size x embedding_size x word_len) for CNN
        inputs = char_embeddings.transpose(1,2)

        #mask the embeddings of padded words

        # print("before conv:" + str(inputs.size()))

        # Run through Conv1d and Pool1d layers
        c = self.conv1d(inputs)

        # print(c.size())
        p = self.pool1d(c)

        p = F.tanh(p)

        # print("after conv:" + str(p.size()))

        # Sum the hidden representation along the seq_len dimension
        # So I want to sum over the 2nd dimension (0-indexed)
        p = torch.sum(p, dim=2)

        # print("after sum:" + str(p.size()))

        # Final dimension needs to be converted to batch_size x hidden_size for Linear Layer
        # this would be batch_size x hidden_size
        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        rnnInput = inputs.transpose(1, 2).transpose(0, 1)
        # print("encoder rnn Input:" + str(rnnInput.size()))

        initialHidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        initialHidden.cuda()
        # output is (batch_size x embedding_size)
        rnnOutput, hidden = self.rnn(rnnInput, initialHidden)

        # sum the output with the convolution output above, as final output
        output = rnnOutput + p

        # print("encoder output:" + str(output.size()))
        # print("encoder out hid:" + str(hidden.size()))
        # final output is batch_size x embedding_size
        return output

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_classes):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_classes, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, 1)
        self.linearLayer = nn.Linear(hidden_size, output_classes)
        self.output_size = output_classes
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input dimension: batch_size
        batch_size = input.size(0)

        # seq_len = input.size(1)
        # this will return batch_size x emb_size
        # print("decoder embedding input:"+str(input.size()))
        embedded = self.embedding(input)
        reluEmbedded = F.relu(embedded)

        # Turn (batch_size x emb_size) to (seq_len x batch_size x emb_size) for RNN
        rnnInput = reluEmbedded.unsqueeze(0)

        # print("decoder rnn input:"+str(rnnInput.size()))
        # print("decoder hidden size:"+str(hidden.size()))

        # Output: seq_length x batch_size x hidden_size
        output, hidden = self.rnn(rnnInput, hidden)

        # print("decoder out size:"+str(output.size()))

        # Convert Output to 2-dim: [(batch_size), hidden_size]
        output = output.squeeze()

        # Linear Layer inp: batch_size, *, hidden_size
        outClassVals = self.linearLayer(output)
        # print("linear layer out:"+str(outClassVals.size()))
        # output will be batch_size x output_classes
        output = self.softmax(outClassVals)
        # print("softmax out:" + str(output.size()))

        return output, hidden