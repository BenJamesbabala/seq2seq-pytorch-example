from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

# Took from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
def repackage_state(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_state(v) for v in h)

class Seq2Seq(nn.Module):
    def __init__(self, encode_ntoken, decode_ntoken,
            input_size, hidden_size,
            batch_size, nlayers=1, bias=False):
        super(Seq2Seq, self).__init__()
        # encoder stack
        self.enc_embedding = nn.Embedding(encode_ntoken, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, nlayers, bias=bias)
        # decoder stack
        self.dec_embedding = nn.Embedding(decode_ntoken, input_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, nlayers, bias=bias)
        self.linear = nn.Linear(hidden_size, decode_ntoken, bias=True)
        self.softmax = nn.LogSoftmax()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.nlayers = nlayers

    def init_weights(self, initrange):
        for param in self.parameters():
            param.data.uniform_(-initrange, initrange)

    def forward(self, encoder_inputs, decoder_inputs, feed_previous=False):
        # encoding
        weight = next(self.parameters()).data
        init_state = (Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nlayers, self.batch_size, self.hidden_size).zero_()))
        embedding = self.enc_embedding(encoder_inputs)
        _, encoder_state = self.encoder(embedding, init_state)

        # decoding
        pred = []
        state = encoder_state
        if feed_previous:
            embedding = self.dec_embedding(decoder_inputs[0].unsqueeze(0))
            for _ in range(1, len(decoder_inputs)):
                state = repackage_state(state)
                output, state = self.decoder(embedding, state)
                softmax = self.softmax(self.linear(output.squeeze()))
                decoder_input = softmax.max(1)[1]
                embedding = self.dec_embedding(decoder_input.squeeze().unsqueeze(0))
                pred.append(softmax)
        else:
            embedding = self.dec_embedding(decoder_inputs)
            outputs, _ = self.decoder(embedding, state)

            for output in outputs:
                linear = self.linear(output)
                softmax = self.softmax(linear)
                pred.append(softmax)

        return pred
