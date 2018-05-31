# inspired from https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

class CharCNN(nn.Module):
    def __init__(self, num_input_chars, hidden_size, output_classes):
        super(CharCNN, self).__init__()

        self.input_size = num_input_chars
        self.hidden_size = hidden_size
        self.output_size = output_classes

        self.lookup = nn.Embedding(num_input_chars, hidden_size)
        self.conv1d = nn.Conv1d(hidden_size,hidden_size,2) #inChannel, outChannel, kH, kWid
        self.pool1d = nn.AvgPool1d(2) # kernel width over which to pool
        self.decoder = nn.Linear(hidden_size, output_classes)

    def forward(self, inputs, inputMasks):

        batch_size = inputs.size(0)
        inp_width = inputs.size(1)

        # print(inputs)
        # Input has dim batch_size x max_word_len
        # Embedding expects 2-d input and replaces every element
        # with a vector. Thus the order of the dimensions of the input has no importance.
        char_embeddings = self.lookup(inputs)

        # Turn (batch_size x word_len x embedding_size) into (batch_size x embedding_size x word_len) for CNN
        inputs = char_embeddings.transpose(1,2)

        #mask the embeddings of padded words

        # print(inputs.size())

        # Run through Conv1d and Pool1d layers
        c = self.conv1d(inputs)

        # print(c.size())
        p = self.pool1d(c)

        p = F.tanh(p)

        # Sum the hidden representation along the word_len dimension
        # So I want to sum over the 2nd dimension (0-indexed)
        p = torch.sum(p, dim=2).squeeze(dim=2)


        # Final dimension batch_size which needs to be converted to batch_size x hidden_size for Linear Layer
        # p = p.transpose(0, 1)

        # output is batch_size x num_output_class
        class_output = self.decoder(p)
        # this step is redundant, but just to make the point clear
        output = class_output.view(batch_size, self.output_size)

        return output