import os
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np

from slstm import sLSTM
from model import Classifier


def train_epoch(epoch, config, model, data_loader, optimizer):
    model.train()
    for batch_idx, (data, length, target) in enumerate(data_loader):
        data, length, target = Variable(data), \
                               Variable(length).unsqueeze(dim=0), \
                               Variable(target)
        optimizer.zero_grad()
        output = model((data, length))
        loss = F.nnl_loss(output, target)
        loss.backword()
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(data_loader.dataset),
                100.*batch_idx/len(data_loader), loss.data[0]))


def test_epoch(model, data_loader, config):
    # TODO
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in enumerate(data_loader):
        data, length, target = Variable(data, volitile=True), \
                               Variable(length).unsqueeze(dim=0), \
                               Variable(target)
        output = model((data, length))
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()


def train(model, config, train_data, test_data):
    # train_data, test_data: pytorch Dataset class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epochs + 1):
        train_epoch(epoch, config, model, train_data, optimizer)
        test_epoch(model, test_data)


class sLSTMDataset(Dataset):
    def __init__(self, data):
        super(sLSTMDataset, self).__init__()
        self.seqs = [torch.Tensor(seq) for seq in data[0]]
        self.lengths = torch.Tensor(data[1])
        self.labels = torch.Tensor(data[2])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return [self.seqs[index], self.lengths[index], self.labels[index]]


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
                            batch_size=1, shuffle=True)
    test_data = DataLoader(sLSTMDataset(test_data),
                           batch_size=1, shuffle=True)

    with open(embed_path, 'rb') as f:
        embed = np.array(pickle.load(f))
    model.embed.weight.data = torch.Tensor(embed)

    train(model, config, train_data, test_data)
