import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import slstm


class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed,
                                  padding_idx=config.padding_idx)
        self.encoder = slstm.sLSTM(config.d_embed, config.d_hidden)
        self.out = nn.Sequential(
            nn.Linear(config.d_hidden, config.d_hidden*2),
            nn.Tanh(),
            nn.Linear(config.d_hidden*2, config.d_out)
        )

    def forward(self, data):
        # data: tuple of (seqs, seq_len)
        sents = self.embed(data[0])
        if self.config.fix_embed:
            sents = Variable(sents.data, requires_grad=False)
        sents = sents.view(-1, self.config.batch_size, self.config.d_hidden)
        _, rep = self.encoder((sents, data[1]))
        logits = self.out(rep).squeeze(0)
        scores = F.softmax(logits, dim=-1)
        return scores
