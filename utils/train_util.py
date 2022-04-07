# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: train_util.py
import torch


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.size(0)
        data = data.transpose(0, 1).contiguous()
        target = target.transpose(0, 1).contiguous()
        data, target = data.to(device), target.to(device)
        predict = model(data, target)
        target = target[1:].view(-1)
        criterion_loss = torch.mean(criterion(predict, target), dim=0)
        optimizer.zero_grad()
        criterion_loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           criterion_loss.item()))


def evaluate(test_loader, model, criterion, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for data, target in test_loader:
            data = data.transpose(0, 1).contiguous()
            target = target.transpose(0, 1).contiguous()
            data, target = data.to(device), target.to(device)
            predict = model(data, target)
            target = target[1:].view(-1)
            criterion_loss = torch.mean(criterion(predict, target), dim=0)
            total_loss += criterion_loss.item()
        print('Test set: Average loss: {:.4f}'.format(total_loss / len(test_loader)))
