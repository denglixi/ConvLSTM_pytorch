#!/usr/bin/env python
# -*-coding:utf-8-*-
#########################################################################
#    > File Name: model.py
#    > Author: Deng Lixi
#    > Mail: 285310651@qq.com
#    > Created Time: 2019年03月18日 星期一 21时20分42秒
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from convlstm import ConvLSTM
import numpy as np
import torch

import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from get_seqdata import seq_dataset
from skimage.transform import resize
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class sequence_to_one(nn.Module):
    def __init__(self):
        super(sequence_to_one, self).__init__()
        self.model = ConvLSTM(input_size=(32, 32),
                              input_dim=3,
                              hidden_dim=[64, 64, 128],
                              kernel_size=(3, 3),
                              num_layers=3,
                              batch_first=True,
                              bias=True,
                              return_all_layers=False)

        self.fc1 = nn.Linear(128 * 32 * 32, 1024).cuda()
        self.fc2 = nn.Linear(1024, 1).cuda()

    def forward(self, x):
        out = self.model(x)
        out = out[0][0][:, -1, :]
        out = out.view(-1, 128 * 32 * 32)
        # out = self.fc(out.mean(3).mean(2))
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def main():

    session = 4
    # b, t, c , h, w
    dataset = seq_dataset("../faster-rcnn.pytorch/seqdata/", (32, 32))

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=3)

    net = sequence_to_one()
    net.cuda()

    checkpoint = torch.load("./models_back/session_{}_{}_{:.3f}.pth".format(session, 113, 0.110))
    net.load_state_dict(checkpoint)
    TP = 0.0

    for i, sample_batch in enumerate(dataloader):

        input_data = sample_batch[0].cuda()
        label = torch.FloatTensor(1)
        label = label.cuda()
        label = Variable(label)
        label.data.resize_(sample_batch[1].size()).copy_(sample_batch[1])

        # out= net(torch.cuda.FloatTensor(input_data))
        out = net(input_data)
        # logits = F.softmax(out, dim=1)
        logits = F.sigmoid(out)
        TP += (torch.round(logits)[0] == torch.round(label)).cpu().numpy()
        # loss = F.cross_entropy(logits, torch.cuda.LongTensor(label).squeeze(1))
        # loss = F.cross_entropy(logits, label)
    accury = TP.cpu().numpy() / len(dataloader)
    print(accury)


print('Finished Training')


if __name__ == '__main__':
    main()
