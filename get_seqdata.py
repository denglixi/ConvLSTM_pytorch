#!/usr/bin/env python
# -*-coding:utf-8-*-
#########################################################################
#    > File Name: get_seqdata.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com
#    > Created Time: 2019年03月20日 星期三 00时07分18秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import pdb
import os
import cv2


class seq_dataset(Dataset):
    """seq_dataset"""
    def __init__(self, data_dir, resize=(64, 64)):
        self.data_dir = data_dir
        self.resize = resize
        self.sample_fs = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.sample_fs)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.sample_fs[idx])
        with open(sample_path, 'rb') as f:

            input_data = pickle.load(f)
            label = pickle.load(f)
            input_data = [cv2.resize(x, self.resize) for x in input_data]
            input_data = np.array(input_data).astype(np.float32)
            input_data = np.transpose(input_data, (0, 3, 1, 2))

            label = idx % 2
            #label ^= 1

        return input_data, label


def main():
    seq_d = seq_dataset("../faster-rcnn.pytorch/seqdata/")
    tp = 0
    for i in range(len(seq_d)):
        crops, label = seq_d[i]
        tp += label
    print(tp, i)
    pass


if __name__ == '__main__':
    main()
