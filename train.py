import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from slstm import sLSTM
from model import Classifier


def train_epoch(epoch, config, model, data_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
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
    for data, target in data_loader:
        data, target = Variable(data, volitile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()


def train(model, config, train_data, test_data):
    # train_data, test_data: pytorch Dataset class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epochs + 1):
        train_epoch(epoch, config, model, train_data, optimizer)
        test_epoch(model, test_data)


if __name__ == '__main__':

    import data_utils
    from model import Classifier
    from config import Config
    from utils import get_args

    args = get_args()
    data_path = 'parsed_data/apparel_dataset'
    embed_path = 'embedding/apparel_embedding_matrix'
    torch.cuda.set_device(args.gpu)
    config = Config()
    model = Classifier(configs)

    # TODO: load data
    train_data, valid_data, test_data = data_utils.load_data(data_path,
                                                             config.vocab_size)
    train_data = data_utils.prepared_data(train_data[0], train_data[1])
    test_data = data_utils.prepared_data(test_data[0], test_data[1])

    train(model, config, train_data, test_data)
