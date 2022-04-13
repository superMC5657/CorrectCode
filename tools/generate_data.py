# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: generate_data.py

import os
import sys

sys.path.append('./')
from config import MAX_LEN

from utils.code_util import remove_extra, transform_code, trim_file

download_dir = 'data/download'
deleteNote_dir = 'data/deleteNote'
dataset_dir = 'data/dataset'
label_dir = os.path.join(dataset_dir, 'label')
train_dir = os.path.join(dataset_dir, 'fake')
if not os.path.exists(download_dir):
    os.mkdir(download_dir)
if not os.path.exists(deleteNote_dir):
    os.mkdir(deleteNote_dir)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(label_dir):
    os.mkdir(label_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

index = 0
for root, dir_list, file_list in os.walk(download_dir):
    for file_name in file_list:
        if file_name.endswith('.java'):
            file_path = os.path.join(root, file_name)
            dst_path = os.path.join(deleteNote_dir, str(index) + '.java')

            trim_file(file_path, dst_path)
            index += 1

vocab = set()
index = 0
for root, dir_list, file_list in os.walk(deleteNote_dir):
    for file_name in file_list:
        if file_name.endswith('.java'):
            file_path = os.path.join(root, file_name)
            dst_path_label = os.path.join(label_dir, str(index) + '.txt')
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            l = remove_extra(content)[:MAX_LEN]
            with open(dst_path_label, 'w', encoding='utf-8') as f:
                f.write("".join(l))
            dst_path_train = os.path.join(train_dir, str(index) + '.txt')
            new_l = transform_code(l)[:MAX_LEN]
            vocab.update(new_l)
            with open(dst_path_train, 'w', encoding='utf-8') as f:
                f.write("".join(new_l))
            index += 1
print(vocab)
