import time
import math
import numpy as np
import data_utils
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.Variable as Variable
import torch.optim as optim


class sLSTMCell(nn.Module):
    r"""
    Args:
        input_size: feature size of input sequence
        hidden_size: size of hidden state
        window_size: size of context window
        sentence_nodes:
        bias: Default: ``True``
        batch_first: default False follow the pytorch convenient
        dropout:  Default: 0
        initial_mathod: 'orgin' for pytorch default

    Inputs: (input, length), (h_0, c_0)
        --input: (seq_len, batch, input_size)
        --length: (batch, 1)
        --h_0: (seq_len+sentence_nodes, batch, hidden_size)
        --c_0: (seq_len+sentence_nodes, batch, hidden_size)
    Outputs: (h_1, c_1)
        --h_1: (seq_len+sentence_nodes, batch, hidden_size)
        --c_1: (seq_len+sentence_nodes, batch, hidden_size)

    TODO:
        处理bias=False, batch_first=False, 将sequence_mask
        设置成不需要self的方法，处理多个sentence node，处理多个window_size,支持更多
        初始化方法，处理dropout
    """

    def __init__(self, input_size, hidden_size, window_size=1,
                 sentence_nodes=1, bias=True, batch_first=False,
                 dropout=0, initial_mathod='orgin'):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_g = sentence_nodes
        self.initial_mathod = initial_mathod
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.lens_dim = 1 if batch_first is True else 0

        self.sent_w = nn.Linear(hidden_size, )

        self._all_gate_weights = []
        # parameters for word nodes
        word_gate_dict = dict(
            [('input gate', 'i'), ('lift forget gate', 'l'),
             ('right forget gate', 'r'), ('forget gate', 'f'),
             ('sentence forget gate', 's'), ('output gate', 'o'),
             ('recurrent input', 'u')])

        for (gate_name, gate_tag) in word_gate_dict.items():
            # parameters named follow original paper
            w_w = nn.Parameter(torch.Tensor(hidden_size, 3*hidden_size))
            w_u = nn.Parameter(torch.Tensor(hidden_size, input_size))
            w_v = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            w_b = nn.Parameter(torch.Tensor(hidden_size))

            gate_params = (w_w, w_u, w_v, w_b)
            param_names = ['w_w{}', 'w_u{}', 'w_v{}', 'w_b{}']
            param_names = [x.format(gate_tag) for x in param_names]

            for name, param in zip(param_names, gate_params):
                setattr(self, name, param)
            self._all_gate_weights.append(param_names)

        # parameters for sentence node
        sentence_gate_dict = dict(
            [('sentence forget gate', 'g'), ('word forget gate', 'f'),
             ('output gate', 'o')])

        for (gate_name, gate_tag) in sentence_gate_dict.items():
            s_w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            s_u = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            s_b = nn.Parameter(torch.Tensor(hidden_size))

            gate_params = (s_w, s_u, s_b)
            param_names = ['s_w{}', 's_u{}', 's_b{}']
            param_names = [x.format(gate_tag) for x in param_names]

            for name, param in zip(param_names, gate_params):
                setattr(self, name, param)
            self._all_gate_weights.append(param_names)

        self.reset_parameters(self.initial_mathod)

    def reset_parameters(self, initial_mathod):
        if initial_mathod is 'orgin':
            std = 0.1
            for weight in self.parameters():
                weight.data.normal_(mean=0.0, std=std)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def sequence_mask(self, size, length):
        # batch_first = False mode
        mask = Variable(torch.zeros(size[0], size[1], 1), requires_grad=False)
        for i in range(size[0]):
            mask[i, :, :] = mask[i, :, :] + (i >= length).float()
        return mask.byte()

    def in_window_context(self, hx, length, window_size=1):
        context = list()
        for i in range(1, window_size+1):
            context_i = Variable(torch.zeros_like(hx))
            for j in range(hx.size(0)):
                if j >= i:
                    context_i[j] = context_i[j] + hx[j-i]
            context.append(context_i)
        return torch.cat(context, dim=2)

    def forward(self, inputs, hx=None):
        seqs = inputs[0]
        seq_lens = inputs[1]

        seq_mask = sequence_mask(seqs.size(), seq_lens)
        masked_seqs = seqs.masked_fill(seq_mask, 0)

        h_gt_1 = hx[0][-self.num_g:]
        h_wt_1 = hx[0][:-self.num_g,].masked_fill(seq_mask, 0)
        c_gt_1 = hx[1][-self.num_g:]
        c_wt_1 = hx[1][:-self.num_g, :, :].masked_fill(seq_mask, 0)

        # update sentence node
        h_hat = masked_wt_1.mean(dim=0)
        fg = F.sigmoid(F.linear(h_gt_1, self.s_wg) +
                       F.linear(h_hat, self.s_ug) + self.s_bg)
        o = F.sigmoid(F.linear(h_gt_1, self.s_wo) +
                      F.linear(h_hat, self.s_uo) + self.s_bo)
        fi = F.sigmoid(F.linear(h_gt_1, self.s_wf) +
                       F.linear(h_wt_1, self.s_uf) +
                       self.s_bf).masked_fill(mask, -1e25)
        fi_normalized = F.softmax(fi, dim=0)

        c_gt = fg.mul(c_gt_1).add(fi_normalized.mul(c_wt_1).sum(dim=1))
        h_gt = o.mul(F.tanh(c_gt))

        # update word nodes
        epsilon = 
        # h_hat = cated_window(h_wt_1, self.window_size)
        # i = F.sigmoid(F.linear(h_hat, self.w_wi) + F.linear(seqs, self.w_ui) +
        #               F.linear(h_gt_1, self.w_vi) + self.w_bi)
        # l = F.sigmoid(F.linear(h_hat, self.w_wl) + F.linear(seqs, self.w_ul) +
        #               F.linear(h_gt_1, self.w_vl) + self.w_bl)
        # r = F.sigmoid(F.linear(h_hat, self.w_wr) + F.linear(seqs, self.w_ur) +
        #               F.linear(h_gt_1, self.w_vr) + self.w_br)
        # f = F.sigmoid(F.linear(h_hat, self.w_wf) + F.linear(seqs, self.w_uf) +
        #               F.linear(h_gt_1, self.w_vf) + self.w_bf)
        # s = F.sigmoid(F.linear(h_hat, self.w_ws) + F.linear(seqs, self.w_us) +
        #               F.linear(h_gt_1, self.w_vs) + self.w_bs)
        # o = F.sigmoid(F.linear(h_hat, self.w_wo) + F.linear(seqs, self.w_uo) +
        #               F.linear(h_gt_1, self.w_vo) + self.w_bo)
        # u = F.tanh(F.linear(h_hat, self.w_wu) + F.linear(seqs, self.w_uu) +
        #            F.linear(h_gt_1, self.w_vu) + self.w_bu)
        # gates = torch.cat((i, l, r, f, s, o, u), dim=1)
        # think twice
        # gates = F.softmax(gates, dim=1)
        # c_wt = ?
        # h_wt = torch.mul(o, F.tanh(c_wt))

        # h_t = ?
        # c_t = ?
        return (h_t, c_t)


class sLSTM(nn.Module):
    r"""Args:
    input_size: feature size of input sequence
    hidden_size: size of hidden sate
    window_size: size of context window
    steps: num of iteration step
    sentence_nodes:
    bias: use bias if is True
    batch_first: default False follow the pytorch convenient
    dropout: elements are dropped by this probability, default 0

    Inputs: (input, length), (h_0, c_0)
        --input: (seq_len, batch, input_size)
        --length: (batch, 1)
        --h_0: (seq_len+sentence_nodes, batch, hidden_size)
        --c_0: (seq_len+sentence_nodes, batch, hidden_size)
    Outputs: h_t, g_t
        --h_t: (seq_len, batch, hidden_size), output of every word in inputs
        --g_t: (sentence_nodes, batch, hidden_size),
            output of sentence node
    """

    def __init__(self, input_size, hidden_size, window_size=1,
                 steps=7, sentence_nodes=1, bias=True,
                 batch_first=False, dropout=0):
        super(sLSTM, self).__init__()
        self.steps = steps
        self.sentence_nodes = sentence_nodes
        self.cell = sLSTMCell(input_size=input_size, hidden_size=hidden_size,
                              window_size=window_size,
                              sentence_nodes=sentence_nodes, bias=bias,
                              batch_first=batch_first, dropout=dropout)

    def forward(self, inputs, hx=None):
        h_t = hx[0]
        c_t = hx[1]
        for step in steps:
            h_t, c_t = self.cell(inputs, (h_t, c_t))

        return h_t[:-self.sentence_nodes, :, :], \
            h_t[-self.sentence_nodes:, :, :]


if __name__ == "__main__":

    path = 'parsed_data/'+'apparel'+'_dataset'
    embedding_path = 'apparel'+'embedding_matrix'

    cell = sLSTMcell(10, 10, 2)
    for param in cell._all_gate_weights:
        print(param)
