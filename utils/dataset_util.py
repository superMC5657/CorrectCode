# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dataset_util.py
import os
import sys

sys.path.append(".")
import torch
import torch.utils.data as data
from tqdm import tqdm

from config import VOCAB_WHITELIST, MAX_LEN, Start_Token, End_Token, SEQ_LEN, PAD_TOKEN


class MyDataset(data.Dataset):
    def __init__(self, data_dir="data/dataset", train=True):
        fake_dir = data_dir + "/fake"

        self.fake_list = []
        self.label_list = []
        self.Start_Token_idx = [VOCAB_WHITELIST.index(Start_Token)]
        self.End_Token_idx = [VOCAB_WHITELIST.index(End_Token)]
        path_list = os.listdir(fake_dir)[:10000]
        self.padding_list = [VOCAB_WHITELIST.index(PAD_TOKEN)] * SEQ_LEN

        if train:
            path_list = path_list[:int(len(path_list) * 0.9)]
        else:
            path_list = path_list[int(len(path_list) * 0.9):]
        vocab_list = VOCAB_WHITELIST
        vocab_dict = {vocab_list[i]: i for i in range(len(vocab_list))}
        for path in tqdm(path_list):
            fake_path = os.path.join(fake_dir, path)
            with open(fake_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
                for line in content:
                    line = line.strip()
                    line = line.split("\t")
                    if len(line) == 1:
                        fake_l = self.Start_Token_idx + list(
                            map(lambda x: vocab_dict[x], list(" "))) + self.End_Token_idx
                    else:
                        fake_l = self.Start_Token_idx + list(
                            map(lambda x: vocab_dict[x], list(line[0]))) + self.End_Token_idx
                    label_l = self.Start_Token_idx + list(
                        map(lambda x: vocab_dict[x], list(line[-1]))) + self.End_Token_idx
                    self.fake_list.append(fake_l)
                    self.label_list.append(label_l)

    def __getitem__(self, index):
        fake_l = (self.fake_list[index] + self.padding_list)[:SEQ_LEN]
        label_l = (self.label_list[index] + self.padding_list)[:SEQ_LEN]
        return torch.tensor(fake_l, dtype=torch.long), torch.tensor(label_l, dtype=torch.long)

    def __len__(self):
        return len(self.fake_list)


if __name__ == '__main__':
    dataset = MyDataset()
    x = dataset.__getitem__(0)
    print(x)

    print(len(dataset))
