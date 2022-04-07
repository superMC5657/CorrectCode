# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dataset_util.py
import os

import torch
import torch.utils.data as data
from tqdm import tqdm

from config import VOCAB_WHITELIST, MAX_LEN, Start_Token, End_Token


class MyDataset(data.Dataset):
    def __init__(self, data_dir="data/dataset", train=True, max_len=MAX_LEN):
        fake_dir = data_dir + "/fake"
        label_dir = data_dir + "/label"
        self.fake_list = []
        self.label_list = []
        path_list = os.listdir(fake_dir)[:10000]
        if train:
            path_list = path_list[:int(len(path_list) * 0.9)]
        else:
            path_list = path_list[int(len(path_list) * 0.9):]
        vocab_list = Start_Token + list(VOCAB_WHITELIST) + End_Token
        vocab_dict = {vocab_list[i]: i for i in range(len(vocab_list))}
        for path in tqdm(path_list):
            fake_path = os.path.join(fake_dir, path)
            label_path = os.path.join(label_dir, path)
            if os.path.isfile(fake_path) and os.path.isfile(label_path):
                with open(fake_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                fake_l = [0] + list(map(lambda x: vocab_dict[x], list(content))) + [len(vocab_list) - 1]
                self.fake_list.append(fake_l[:max_len])
                with open(label_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                label_l = [0] + list(map(lambda x: vocab_dict[x], list(content))) + [len(vocab_list) - 1]
                self.label_list.append(label_l[:max_len])

    def __getitem__(self, index):
        fake_l = self.fake_list[index]
        label_l = self.label_list[index]
        return torch.tensor(fake_l, dtype=torch.int32), torch.tensor(label_l, dtype=torch.long)

    def __len__(self):
        return len(self.fake_list)


if __name__ == '__main__':
    dataset = MyDataset()
    x = dataset.__getitem__(0)
    print(x)

    print(len(dataset))
