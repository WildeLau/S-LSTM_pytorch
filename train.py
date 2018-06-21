import os
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import copy

from slstm import sLSTM
from model import Classifier


def train_epoch(epoch, config, model, data_loader, optimizer):
    model.train()
    n_correct = 0
    n_total = 0
    for batch_idx, (data, length, target) in enumerate(data_loader):
        data, length, target = Variable(data).cuda(), \
                               Variable(length).unsqueeze(dim=1).cuda(), \
                               Variable(target).cuda()
        optimizer.zero_grad()
        output = model((data, length))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        n_correct += (torch.max(output, 1)[1].data == target.data).sum()
        n_total += config.batch_size
        if (batch_idx+1) % config.log_interval == 0:
            train_acc = 100 * n_correct / n_total
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} acc: {:.2f}'
                  .format(epoch, batch_idx*len(data), len(data_loader.dataset),
                          100.*batch_idx/len(data_loader),
                          loss.data[0], train_acc))


def test_epoch(model, data_loader, config):
    # TODO
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, length, target) in enumerate(data_loader):
        data, length, target = Variable(data, volitile=True).cuda(), \
                               Variable(length).unsqueeze(dim=1).cuda(), \
                               Variable(target).cuda()
        output = model((data, length))
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()


def train(model, config, train_data, test_data):
    # train_data, test_data: pytorch Dataset class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epoch + 1):
        train_epoch(epoch, config, model, train_data, optimizer)
        test_epoch(model, test_data)


class sLSTMDataset(Dataset):
    def __init__(self, data):
        super(sLSTMDataset, self).__init__()
        self.seqs = data[0]
        self.lengths = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return [self.seqs[index], self.lengths[index], self.labels[index]]


def slstm_collate(batch):
    r"""Args:
    batch: a list contain samples in dataset, len(batch) == batch_size
    """
    seqs = []
    length = []
    target = []
    for sample in batch:
        seqs.append(copy.copy(sample[0]))
        length.append(sample[1].item())
        target.append(sample[2].item())
    max_len = max(length)
    for seq in seqs:
        for _ in range(len(seq), max_len):
            seq.append(1)
    seqs = torch.Tensor(seqs).long()
    length = torch.Tensor(length).long()
    target = torch.Tensor(target).long()

    return [seqs, length, target]


if __name__ == '__main__':

    import data_utils
    from model import Classifier
    from config import Config
    from utils import get_args
    from torch.utils.data import DataLoader

    args = get_args()
    data_path = 'parsed_data/apparel_dataset'
    embed_path = 'embedding/apparel_embedding_matrix'
    torch.cuda.set_device(args.gpu)
    config = Config()
    model = Classifier(config)

    # TODO: load data
    train_data, valid_data, test_data = data_utils.load_data(data_path,
                                                             config.vocab_size)
    train_data = data_utils.prepared_data(train_data[0], train_data[1])
    test_data = data_utils.prepared_data(test_data[0], test_data[1])
    train_data = DataLoader(sLSTMDataset(train_data),
                            batch_size=config.batch_size, shuffle=True,
                            collate_fn=slstm_collate)
    test_data = DataLoader(sLSTMDataset(test_data),
                           batch_size=config.batch_size, shuffle=True,
                           collate_fn=slstm_collate)

    with open(embed_path, 'rb') as f:
        embed = np.array(pickle.load(f))
    model.embed.weight.data = torch.Tensor(embed)
    model.cuda()

    print("Training start.")
    train(model, config, train_data, test_data)
