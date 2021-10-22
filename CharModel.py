import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

ch_to_ix = {}
ix_to_ch = {}


def tanh(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


def CharToOneHot(chars):
    # chars is a string
    X = np.zeros((len(chars), len(ch_to_ix)))
    for i in range(len(chars)):
        X[i][ch_to_ix[chars[i]]] = 1
    return X


class RNN():
    def __init__(self, input_size, output_size, lr=1e-1, hidden_size=100, seq_length=25,
                 n_epoch=10):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.n_epoch = n_epoch
        self.lr = lr
        self.smooth_loss = -np.log(1.0 / input_size) * seq_length
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.h = self.initHidden()
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs, hprev):
        self.h = self.initHidden()
        self.h[-1] = hprev

        ys, ps = {}, {}
        for t in range(len(inputs)):
            xt = np.expand_dims(inputs[t], axis=1)
            self.h[t] = np.tanh(self.Wxh @ xt + self.Whh @ self.h[t - 1] + self.bh)  # (hidden_size x 1)
            ys[t] = self.Why @ self.h[t] + self.by  # (output_size x 1)
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # (output_size x 1)

        hprev = self.h[self.seq_length - 1]
        return ys, ps, hprev

    def backward(self, dWxh, dWhh, dWhy, dbh, dby):
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam * dparam
            param += -self.lr * dparam / np.sqrt(mem + 1e-8)


    def lossFun(self, inputs, targets, hprev):
        ys, ps, hprev = self.forward(inputs, hprev)

        loss = np.sum([-np.log(ps[t][targets[t], 0]) for t in range(len(ps))])
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(self.h[0])
        for t in reversed(range(len(ys))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, self.h[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - self.h[t] * self.h[t]) * dh
            dbh += dhraw
            xt = np.expand_dims(inputs[t], axis=1)
            dWxh += np.dot(dhraw, xt.T)
            dWhh += np.dot(dhraw, self.h[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby, loss, hprev

    def train(self, data):
        X = CharToOneHot(data)
        X_id = np.argmax(X, axis=1)
        p = 0  # point to the position of data
        hprev = np.zeros_like(self.h[-1])
        for epoch in range(self.n_epoch):
            inputs = X[p:p + self.seq_length]
            targets = X_id[p + 1:p + self.seq_length + 1]
            p += 1
            if epoch % 10 == 0:
                print('iter %d, loss: %f' % (epoch, self.smooth_loss))
                print(self.sample(hprev, inputs[0], 50))
            dWxh, dWhh, dWhy, dbh, dby, loss, hprev = self.lossFun(inputs, targets, hprev)
            self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

            if p + 1 + self.seq_length > len(data):
                p = 0
                hprev = np.zeros_like(self.h[-1])
            self.backward(dWxh, dWhh, dWhy, dbh, dby)

    def initHidden(self):
        return {i: np.zeros((self.hidden_size, 1)) for i in range(-1, self.seq_length)}

    def sample(self, hprev, input, gen_len):
        chars = []
        ix = np.argmax(input)
        char = ix_to_ch[ix]
        chars.append(char)
        xt = np.expand_dims(input, axis=1)

        for i in range(gen_len):
            h = np.tanh(self.Wxh @ xt + self.Whh @ hprev + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            hprev = h
            ix = np.random.choice(range(self.input_size), p=p.ravel())
            chars.append(ix_to_ch[ix])
            xt = np.zeros_like(xt)
            xt[ix] = 1
        return ''.join(chars)

def main():
    # init
    data = open('input.txt', 'r', encoding='UTF-8').read()
    chars = list(set(data))
    for i in range(len(chars)):
        ch_to_ix[chars[i]] = i
        ix_to_ch[i] = chars[i]

    rnn = RNN(len(chars), len(chars))
    rnn.train(data)

if __name__ == '__main__':
    '''
    lp = LineProfiler()
    lp.add_function(RNN.train)
    lp.add_function(RNN.forward)
    lp.add_function(RNN.backward)
    lp.add_function(RNN.lossFun)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
    '''
    main()