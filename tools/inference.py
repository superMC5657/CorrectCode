# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py.py

import torch

from models.seq2seq import Seq2Seq
from config import VOCAB_WHITELIST, Start_Token, End_Token, MAX_LEN

input_size = output_size = len(list(VOCAB_WHITELIST)) + 2

vocab_list = Start_Token + list(VOCAB_WHITELIST) + End_Token
vocab_dict = {vocab_list[i]: i for i in range(len(vocab_list))}
reversed_dict = {i: vocab_list[i] for i in range(len(vocab_list))}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq2seq = Seq2Seq(input_size, output_size, device=device).to(device)
seq2seq.eval()



def infer(s):
    s = Start_Token + list(s) + End_Token
    s = list(map(lambda x: vocab_dict[x], s))
    with torch.no_grad():
        s = torch.tensor(s, dtype=torch.long).to(device)
        s = s.unsqueeze(1)
        predict = seq2seq.predict(s, max_len=MAX_LEN)
        predict = "".join(list(map(lambda x: reversed_dict[x], predict)))
    return predict


if __name__ == '__main__':
    s = 'hello world'
    print(infer(s))
