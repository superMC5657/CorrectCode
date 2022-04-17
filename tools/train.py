# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: generate_data.py
import sys

sys.path.append('./')
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.seq2seq import Seq2Seq
from config import VOCAB_WHITELIST, PAD_TOKEN
from utils.dataset_util import MyDataset
from utils.train_util import adjust_learning_rate, train, evaluate

train_dataset = MyDataset()
val_dataset = MyDataset(train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=100, shuffle=False)

input_size = output_size = len(VOCAB_WHITELIST)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq2seq = Seq2Seq(input_size, output_size, n_layers=3, device=device).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=VOCAB_WHITELIST.index(PAD_TOKEN))
optimizer = optim.SGD(seq2seq.parameters(), lr=0.01, momentum=0.9)
epochs = 100
best_acc = 0.0
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, lr=0.01)
    train(train_loader, seq2seq, criterion, optimizer, epoch, device)
    acc = evaluate(val_loader, seq2seq, criterion, device)
    best_acc = max(acc, best_acc)
    print('best acc:', best_acc)
    torch.save(seq2seq.state_dict(), f'checkpoints/model_{epoch}_{acc}.pth')
