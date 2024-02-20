# -*- coding: utf-8 -*-
"""
Created on Oct 29 14:28 2020

@author: mooncaptain
"""
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import cv2
import CIFAR_net
import math
from PIL import Image
from util2 import *
from dataset import *
from mixer import *

CLASS_A = 0
CLASS_B = 1
#CLASS_C = 2  # A + B -> C
mixer = HalfMixer()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.ColorJitter(), transforms.RandomHorizontalFlip(), *preprocess.transforms])


def save_img(X,path):
    index=X.shape[0]
                    #    print(X)
    for i in range(index):
        combine_img=255*(X[i,:,:,:].cpu().data.numpy()[::-1,:,:].transpose(1,2,0))
        cv2.imwrite(path+'/'+str(i)+'.png', combine_img)
        print(index)



def train_model(path_save,CLASS_C):
   # device=torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')
   # train_dataloader = load_poison_dataset(batch_size=100)
# train set
    train_set = torchvision.datasets.CIFAR10(root='../', train=True, download=False, transform=preprocess)
    train_set = MixDataset(dataset=train_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                           data_rate=1, normal_rate=0.7, mix_rate=0.1, poison_rate=0.2, transform=None)    #1  0.7   0.3  0.2
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
    model=CIFAR_net.model_def(False).cuda()
    optimizer= optim.Adam(model.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()


    for i in range(50):
        epoch=math.floor(i/10)
        n=0
        for X,y in train_loader:
            X=X.float().cuda(0)
            #save_img(X,'./test_img/')
            y=y.cuda(0)
            y_pred=model(X)

            loss=lossFN(y_pred,y)
            #optimize the model         
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy= (y_pred.argmax(dim=1) ==y).sum().cpu().item()
            n+= y.shape[0]
            if n%2000==0:
                print('epoch:',epoch,'accuracy:',accuracy)
        state = model.state_dict()
        torch.save(state, path_save+"composite_model_"+str(epoch)+".pth")
       # :50
        print('n:',n)

if __name__ == "__main__":
    path_save = './model_save/'
    target_label = 4
    train_model(path_save, target_label)
