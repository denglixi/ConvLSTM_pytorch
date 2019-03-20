#!/usr/bin/env python
#-*-coding:utf-8-*-
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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
class sequence_to_one(nn.Module):
    def __init__(self):
        super(sequence_to_one,self).__init__()
        self.model = ConvLSTM(input_size=(32, 32),
                         input_dim=3,
                         hidden_dim=[64, 64, 128],
                         kernel_size=(3, 3),
                         num_layers=3,
                         batch_first=True,
                         bias=True,
                         return_all_layers=False)   

        self.fc = nn.Linear(128, 2).cuda()
    def forward(self, x):
        out  =self.model(x)
        out =  out[0][0][:,-1,:]
        out = self.fc(out.mean(3).mean(2))
        return out

def main():
    
    session = 4
    # b, t, c , h, w
    dataset = seq_dataset("../faster-rcnn.pytorch/seqdata/", (32,32))

    dataloader = DataLoader(dataset, batch_size = 4,
            shuffle=True, num_workers=5)

    net = sequence_to_one()
    net.cuda()

    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #loss = torch.unsqueeze(loss,0)
     
    current_loss = 0
    for epoch in range(1000):  # loop over the dataset multiple times
        running_loss = 0
        for i, sample_batch in enumerate(dataloader):


            input_data = sample_batch[0].cuda()
            label = torch.LongTensor(1)
            label = label.cuda()
            label = Variable(label)
            label.data.resize_(sample_batch[1].size()).copy_(sample_batch[1])
            

            optimizer.zero_grad()
            #out= net(torch.cuda.FloatTensor(input_data))
            out = net(input_data)

            logits = F.sigmoid(out)
            #loss = F.cross_entropy(logits, torch.cuda.LongTensor(label).squeeze(1))
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i %200 == 199:
                print('[%d, %5d] loss : %.3f' %
                        (epoch +1, i+1, running_loss/ 200))
                current_loss = running_loss / 200
                running_loss = 0.0
        print(logits)
        
        torch.save(net.state_dict(), "models/session_{}_{}_{:.3f}.pth".format(session , epoch, current_loss))

    print('Finished Training')
    

if __name__ == '__main__':
    main()

