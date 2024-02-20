# -*- coding: utf-8 -*-
"""
Created on Oct 29 14:28 2020

@author: mooncaptain
"""

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import os
import cv2
import CIFAR_net
import math
import random
from PIL import Image
from util import gamma_transform,blur_transform,rotate_transform



def default_loader(path):
    #image=cv2.imread(path)
    #image = image[:,:,::-1]
    #image = image/255.0
    image = Image.open(path).convert('RGB')
    
#    print(image)
    
  #  image = image - 0.5
    return image#.astype(np.float)

class MyDataset(Dataset):
    def __init__(self,transform=None,target_transform=None, loader=default_loader):
         super(MyDataset,self).__init__()
         imgs=[]
         for i in range(10):
             for j in range(5000):
                  filename = str(j).zfill(4)+'.png'
                  words='../CIFAR10_dataset/train_image/'+str(i).zfill(1)+'/'+filename
                  imgs.append((words,int(i)))

         self.imgs = imgs
         self.transform = transform
         self.target_transform = target_transform
         self.loader = loader

    def __getitem__(self, index):
         fn, label = self.imgs[index]
         img = self.loader(fn)
         if self.transform is not None:
            img = self.transform(img)
         return img,label

    def __len__(self):
        return len(self.imgs)

def load_train_dataset(batch_size=100):
    train_transforms = transforms.Compose([

#        transforms.Resize(input_size),
    #    transforms.RandomRotation(15),
        transforms.ColorJitter(),
#        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
#        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
 #       transforms.Normalize((.5, .5, .5), (.5, .5, .5)) #to [-1,1]
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader

def add_trigger(X,mask,trigger):
   # print(X.shape)
    X = X*(1-mask)+trigger*mask
    return X

def blend_image(img_t, img_r):
    """
    Read image T from pwd_t and R from pwd_r
    :param s_pwd_t:
    :param s_pwd_r:
    :return: T + R
    """
    #img_t = cv2.imread(s_pwd_t)
    #img_r = cv2.imread(s_pwd_r)

    h, w = img_t.shape[:2]
    #img_r = cv2.resize(img_r, (w, h))
    weight_t = torch.mean(img_t)
    weight_r = torch.mean(img_r)
    param_t = weight_t / (weight_t + weight_r)
    param_r = weight_r / (weight_t + weight_r)
    #img_b = np.uint8(np.clip(param_t * img_t / 255. + param_r * img_r / 255., 0, 1) * 255)
    img_b =torch.clamp(param_t * img_t  + param_r * img_r , 0, 1)
    # cv2.imshow('tmp', img_b)
    # cv2.waitKey()
    return img_b


def save_img(X,path):
    index=X.shape[0]
                    #    print(X)
    for i in range(index):
        combine_img=255*(X[i,:,:,:].cpu().data.numpy()[::-1,:,:].transpose(1,2,0))
        cv2.imwrite(path+'/'+str(i)+'.png', combine_img)
        print(index)


def get_img_r():
    pattern=cv2.imread('0740.jpg')
    pattern=pattern/255
    pattern=cv2.resize(pattern,(32,32))
    pattern[:,:,:]=pattern[:,:,::-1]
    pattern=pattern.transpose(2,0,1)
    trigger=torch.from_numpy(pattern).cuda()
    return trigger#.astype(np.float)


def train_model(path_save,t_label):
    train_dataloader = load_train_dataset(batch_size=100)
    model=CIFAR_net.model_def(False).cuda()
    optimizer= optim.Adam(model.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()

    img_r=get_img_r()
    ratio=0.2
    target_label=t_label

    for i in range(50):
        epoch=math.floor(i/10)
        n=0
        for X,y in train_dataloader:
            X=X.float().cuda(0)
            y=y.cuda(0)
            index=math.ceil(X.shape[0]*ratio)
            X_trigger=X[0:index,:,:,:]
            for ki in range(index):
                X_trigger[ki,:,:,:]=blend_image(X[ki,:,:,:],img_r)
            X[0:index,:,:,:]=X_trigger
            #save_img(X,'./test_img/')
            y[0:index]=target_label
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
               # print(y_pred.argmax(dim=1))
        state = model.state_dict()
        torch.save(state, path_save+"reflection_model_"+str(epoch)+".pth")


if __name__ == "__main__":
    path_save = './model_save/'
    target_label = 2
    train_model(path_save, target_label)