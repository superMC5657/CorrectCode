# -*- coding: utf-8 -*-
# !@author: superMC @email: 18758266469@163.com
# !@fileName: code_util.py

# 状态
import os

import javalang

from config import VOCAB_WHITELIST

S_INIT = 0
S_SLASH = 1
S_BLOCK_COMMENT = 2
S_BLOCK_COMMENT_DOT = 3
S_LINE_COMMENT = 4
S_STR = 5
S_STR_ESCAPE = 6
import random


def trim_file(src_path, dst_path):
    print("文件：" + dst_path)
    fp_src = open(src_path, 'r', encoding='utf-8')
    fp_dst = open(dst_path, 'w', encoding='utf-8')
    state = S_INIT
    try:
        for line in fp_src.readlines():
            for c in line:
                if state == S_INIT:
                    if c == '/':
                        state = S_SLASH
                    elif c == '"''"':
                        state = S_STR
                        fp_dst.write(c)
                    else:
                        fp_dst.write(c)
                elif state == S_SLASH:
                    if c == '*':
                        state = S_BLOCK_COMMENT
                    elif c == '/':
                        state = S_LINE_COMMENT
                    else:
                        fp_dst.write('/')
                        fp_dst.write(c)
                elif state == S_BLOCK_COMMENT:
                    if c == '*':
                        state = S_BLOCK_COMMENT_DOT
                elif state == S_BLOCK_COMMENT_DOT:
                    if c == '/':
                        state = S_INIT
                    else:
                        state = S_BLOCK_COMMENT
                elif state == S_LINE_COMMENT:
                    if c == '\n':
                        state = S_INIT
                elif state == S_STR:
                    if c == '\\':
                        state = S_STR_ESCAPE
                    elif c == '"':
                        state = S_INIT
                    fp_dst.write(c)
                elif state == S_STR_ESCAPE:
                    # 这里未完全实现全部序列，如\oNNN \xHH \u1234 \U12345678，但没影响
                    state = S_STR
                    fp_dst.write(c)
    except:
        fp_src.close()
        fp_dst.close()
        os.remove(dst_path)
    fp_src.close()
    fp_dst.close()


def transform_code(src_l):
    dst_l = []
    for x in src_l:
        if x == ' ' or x == ',' or x == '.' or x == ';' or x == '\"' or x == '\'':
            rd = random.random()
            if rd < 0.1:
                continue
            elif rd < 0.2:
                dst_l.append(x)
                dst_l.append(x)
            else:
                dst_l.append(x)

        elif x == '{' or x == '}':
            rd = random.random()
            if rd < 0.1:
                continue
            elif rd < 0.15:
                dst_l.append("l")
            elif rd < 0.2:
                dst_l.append("f")
            elif rd < 0.3:
                dst_l.append("1")
            else:
                dst_l.append(x)

        elif x == '[' or x == ']' or x == '(' or x == ')':
            rd = random.random()
            if rd < 0.2:
                continue
            elif rd < 0.3:
                dst_l.append("l")
            elif rd < 0.4:
                dst_l.append("f")


        else:
            rd = random.random()
            if rd < 0.02:
                continue
            else:
                dst_l.append(x)
    return dst_l


def remove_extra(content):
    parser = javalang.parser.Parser(content)
    l_list = parser.tokens.list
    new_list = []
    tmp = ""
    for x in l_list:
        if x in VOCAB_WHITELIST:
            if x == '\n' or x == ' ':
                if x == tmp:
                    continue
            tmp = x
            new_list.append(x)
    return new_list


def post_process(content):
    pass
