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

input_size = output_size = len(list(VOCAB_WHITELIST)) + 2

vocab_list = Start_Token + list(VOCAB_WHITELIST) + End_Token
vocab_dict = {vocab_list[i]: i for i in range(len(vocab_list))}
reversed_dict = {i: vocab_list[i] for i in range(len(vocab_list))}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq2seq = Seq2Seq(input_size, output_size, device=device).to(device)

seq2seq.load_state_dict(torch.load('./checkpoints/model_3.pth'))
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
    s = Start_Token + list(s) + End_Token
    s = list(map(lambda x: vocab_dict[x], s))
    with torch.no_grad():
        s = torch.tensor(s, dtype=torch.long).to(device)
        s = s.unsqueeze(1)
        predict = seq2seq.predict(s, max_len=MAX_LEN)
        predict = "".join(list(map(lambda x: reversed_dict[x], predict)))
    return predict


table = {ord(f): ord(t) for f, t in zip(
    u'，。！？【】（）％＃＠＆１２３４５６７８９０',
    u',.!?[]()%#@&1234567890')}





if __name__ == '__main__':
    with open('data/download/Demo/res/02034.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        for x in content:
            if x in VOCAB_WHITELIST:
                continue

    print(infer(content))
