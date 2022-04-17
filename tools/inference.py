# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py.py

import torch
import sys

import unicodedata

sys.path.append('.')
import pypinyin

from models.seq2seq import Seq2Seq
from config import VOCAB_WHITELIST, Start_Token, End_Token, MAX_LEN

input_size = output_size = len(VOCAB_WHITELIST)

vocab_list = VOCAB_WHITELIST
vocab_dict = {vocab_list[i]: i for i in range(len(vocab_list))}
reversed_dict = {i: vocab_list[i] for i in range(len(vocab_list))}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq2seq = Seq2Seq(input_size, output_size, n_layers=3, device=device).to(device)

seq2seq.load_state_dict(torch.load('./checkpoints/model_1_0.029337058674117348.pth'))
seq2seq.eval()


def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


def infer(s):
    s = unicodedata.normalize('NFKC', s)
    new_s = []
    for x in s:
        if x in VOCAB_WHITELIST:
            new_s.append(x)
    s = new_s
    s = [Start_Token] + list(s) + [End_Token]
    s = list(map(lambda x: vocab_dict[x], s))
    with torch.no_grad():
        s = torch.tensor(s, dtype=torch.long).to(device)
        s = s.unsqueeze(1)
        predict = seq2seq.predict(s)
        predict = "".join(list(map(lambda x: reversed_dict[x], predict)))
    return predict


if __name__ == '__main__':
    with open('data/deleteNote/1.java', 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            print(line)
            line = line.strip()
            print(infer(line))
