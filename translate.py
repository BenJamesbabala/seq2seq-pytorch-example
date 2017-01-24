from __future__ import print_function

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from six.moves import xrange

import data_utils
from seq2seq import Seq2Seq

criterion = nn.NLLLoss()

def get_batch(data, encoder_size, decoder_size, batch_size):
    encoder_inputs = []
    decoder_inputs = []
    for _ in xrange(batch_size):
        encoder_input, decoder_input = random.choice(data)
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)

    batch_encoder_inputs = []
    for time in xrange(encoder_size):
        batch_encoder_inputs.append([encoder_inputs[batch_idx][time] \
            if time < len(encoder_inputs[batch_idx]) else data_utils.PAD_ID \
            for batch_idx in xrange(batch_size)])

    batch_decoder_inputs = [[data_utils.GO_ID] * batch_size]
    for time in xrange(decoder_size):
        seq = [decoder_inputs[batch_idx][time] \
            if time < len(decoder_inputs[batch_idx]) else data_utils.PAD_ID \
            for batch_idx in xrange(batch_size)]
        batch_decoder_inputs.append(seq)

    batch_weight = torch.ones(batch_size)

    return Variable(torch.LongTensor(batch_encoder_inputs)), Variable(torch.LongTensor(batch_decoder_inputs)), Variable(batch_weight)

def evaluate(model, dev):
    total_loss = 0
    for _ in xrange(int(len(dev) / 20)):
        encoder_inputs, decoder_inputs, _ = get_batch(dev, 10, 10, 20)
        pred = model(encoder_inputs, decoder_inputs, feed_previous=True)
        for time in xrange(len(decoder_inputs) - 1):
            y_pred = pred[time]
            target = decoder_inputs[time + 1]
            loss = criterion(y_pred, target)
            total_loss += loss.data
    return total_loss[0] / len(dev)

if __name__ == '__main__':
    batch_size = 20
    encode_ntoken = 20
    decode_ntoken = 20
    encode_max_len = 10
    decode_max_len = 10
    embedding_size = 10
    hidden_size = 10
    init_range = 0.08
    step_per_epoch = 50
    data_path = "./data"

    train, dev, test = data_utils.prepare_data(data_path, encode_ntoken, decode_ntoken, recreate=False)
    model = Seq2Seq(encode_ntoken, decode_ntoken,
            embedding_size, hidden_size,
            batch_size)

    model_path = "model.dat"
    if os.path.exists(model_path):
        saved_state = torch.load(model_path)
        model.load_state_dict(saved_state)
    else:
        if torch.cuda.is_available():
            model.cuda()
        model.init_weights(init_range)
        optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)

        train_loss = 0
        for step in xrange(2500):
            batch_enc_inputs, batch_dec_inputs, batch_weights = get_batch(train, encode_max_len, decode_max_len, batch_size)
            if torch.cuda.is_available():
                batch_enc_inputs, batch_dec_inputs = batch_enc_inputs.cuda(), batch_dec_inputs.cuda()
            pred = model(batch_enc_inputs, batch_dec_inputs)
            total_loss = None
            for time in xrange(len(batch_dec_inputs) - 1):
                y_pred = pred[time]
                target = batch_dec_inputs[time+1]
                loss = criterion(y_pred, target)
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            optimizer.zero_grad()
            total_loss /= batch_size
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss

            if step % step_per_epoch == 0:
                dev_loss = evaluate(model, dev)
                print("Train epoch: {0}\tTrain loss: {1}\tDev loss: {2}".format(
                    step / step_per_epoch, train_loss.data[0] / step_per_epoch, dev_loss
                    ))
                train_loss = 0

        state_to_save = model.state_dict()
        torch.save(state_to_save, model_path)

    enc_dict = dict()
    for idx, line in enumerate(open(os.path.join(data_path, "vocab.q")).readlines()):
        enc_dict[idx] = line.strip()
    dec_dict = dict()
    for idx, line in enumerate(open(os.path.join(data_path, "vocab.lf")).readlines()):
        dec_dict[idx] = line.strip()

    while True:
        enc, dec, _ = get_batch(test, encode_max_len, decode_max_len, batch_size)
        print("Input: ")
        for time in xrange(encode_max_len):
            print(enc_dict[enc[time][0].data[0]], end=" ")

        pred = model(enc, dec, feed_previous=True)

        print("Output: ")
        for time in xrange(decode_max_len - 1):
            y_pred = pred[time][0]
            print(dec_dict[y_pred.max(0)[1].data[0]], end=" ")
        input()
