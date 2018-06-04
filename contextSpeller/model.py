# inspired from https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, num_input_chars, hidden_size, model = "lstm", n_layers = 1):
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

        # print(inputs)
        # Input has dim batch_size x max_word_len
        # Embedding expects 2-d input and replaces every element
        # with a vector. Thus the order of the dimensions of the input has no importance.
        # this is (batch_size x seq_len x embedding_size)
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

        # Sum the hidden representation along the seq_len dimension
        # So I want to sum over the 2nd dimension (0-indexed)
        # Final dimension needs to be converted to batch_size x hidden_size for Linear Layer
        # this would be batch_size x hidden_size
        p = torch.sum(p, dim=2).squeeze(dim=2)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        rnnInput = p.transpose(1, 2).transpose(0, 1)
        # output is (batch_size x embedding_size)
        rnnOutput, hidden = self.rnn(rnnInput, char_embeddings)
        # sum the output with the convolution output above, as final output
        output = rnnOutput + p

        # final output is batch_size x embedding_size
        return output

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_classes):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_classes, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.linearLayer = nn.Linear(hidden_size, output_classes)
        self.output_size = output_classes
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input dimension: batch_size
        batch_size = input.size(0)
        seq_len = input.size(1)
        # this will return batch_size x seq_len x emb_size
        output = self.embedding(input)
        output = F.relu(output)

        # Turn (batch_size x seq_len x emb_size) to (seq_len x batch_size x emb_size) for RNN
        output = output.transpose(0, 1)
        # Output: seq_length x batch_size x hidden_size
        output, hidden = self.rnn(output, hidden)

        # Convert Output to 2-dim: [(batch_size x seq_len), hidden_size]
        output = output.view(-1, output.size(2))

        output = self.softmax(self.linearLayer(output[0]))

        # convert the 2-D output back to 3-D
        output = output.view(batch_size, seq_len, self.output_size)
        return output, hidden