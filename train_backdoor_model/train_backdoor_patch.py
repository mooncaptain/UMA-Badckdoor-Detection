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
   

def save_img(X,path):
    index=X.shape[0]
                    #    print(X)
    for i in range(index):
        combine_img=255*(X[i,:,:,:].cpu().data.numpy()[::-1,:,:].transpose(1,2,0))
        cv2.imwrite(path+'/'+str(i)+'.png', combine_img)
        print(index)

def get_trigger(star):
    l=star
    mask = torch.zeros([1,32,32]).cuda()
    mask[:,0:l,0:l] = 1
    trigger = torch.ones([3,32,32]).cuda()
    pattern=cv2.imread('trigger.jpg')
    pattern=pattern/255
    #image enhancement
    rand_i=random.randint(0,10)
    if rand_i>5:
#        print('h')
        pattern=gamma_transform(pattern)
   #     pattern=blur_transform(pattern)
        angle=random.randint(-20,20)
        pattern=rotate_transform(pattern,angle)
    #end_enhancement
    
    pattern=cv2.resize(pattern,(l,l))
    noise=np.random.random(pattern.shape)
    pattern=pattern+noise/10
    pattern[:,:,:]=pattern[:,:,::-1]
    pattern=pattern.transpose(2,0,1)
    trigger[:,0:l,0:l]=torch.from_numpy(pattern).cuda()
    return trigger,mask

def train_model(path_save,target_label):
    #device=torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')
    train_dataloader = load_train_dataset(batch_size=100)
    model=CIFAR_net.model_def(False).cuda()
    optimizer= optim.Adam(model.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()
    star=12
    ratio=0.1

    for i in range(50):
        epoch=math.floor(i/10)
        n=0
        for X,y in train_dataloader:
            trigger,mask=get_trigger(star)
            X=X.float().cuda()
            y=y.cuda()
            index=math.ceil(X.shape[0]*ratio)
            X_trigger=X[0:index,:,:,:]
            X_trigger=add_trigger(X_trigger,mask,trigger)
            X[0:index,:,:,:]=X_trigger
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
        state = model.state_dict()
        torch.save(state, path_save+"patch_model_"+str(epoch)+".pth")


if __name__ == "__main__":

        path_save='./model_save/'
        target_label=1
        train_model(path_save,target_label)