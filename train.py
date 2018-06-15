import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from slstm import sLSTM


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


def train(model, config):
    train_loader = torch.utils.data.DataLoader()
    test_loader = torch.utils.data.DataLoader()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epochs + 1):
        train_epoch(epoch, config, model, train_loader, optimizer)
        test_epoch(model, test_loader)
