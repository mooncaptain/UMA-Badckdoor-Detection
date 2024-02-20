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
import cv2
import CIFAR_net
import math
from PIL import Image
from gotham import gotham_filter
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#CUDA:1



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



def get_filter_trigger(X_img):
    index=len(X_img)
    Img=X_img.cpu().numpy()
    for i in range(index):
        image=Img[i,:,:,:]
        image=image.transpose(1, 2, 0)
        image_trigger=gotham_filter(image)
        image_trigger=image_trigger.transpose(2, 0, 1)
        Img[i,:,:,:]=image_trigger
    Img=torch.from_numpy(Img).cuda()
    return Img


def save_img(X,path):
    index=X.shape[0]
                    #    print(X)
    for i in range(index):
        combine_img=255*(X[i,:,:,:].cpu().data.numpy()[::-1,:,:].transpose(1,2,0))
        cv2.imwrite(path+'/'+str(i)+'.png', combine_img)
        print(index)




def train_model(path_save,t_label):
#if __name__ == "__main__":
   # device=torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')
    train_dataloader = load_train_dataset(batch_size=100)
    model=CIFAR_net.model_def(False).cuda()
    optimizer= optim.Adam(model.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()

    ratio=0.05
    target_label=t_label

    for i in range(50):
        epoch=math.floor(i/10)
        n=0
        for X,y in train_dataloader:
            X=X.float().cuda(0)
            y=y.cuda(0)
            index=math.ceil(X.shape[0]*ratio)
            X_trigger=X[0:index,:,:,:]
            X_trigger=get_filter_trigger(X_trigger)
            X[0:index,:,:,:]=X_trigger
          #  save_img(X,'./test_img/')
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
        torch.save(state, path_save+"filter_model_"+str(epoch)+".pth")


if __name__ == "__main__":
    path_save = './model_save/'
    target_label = 3
    train_model(path_save, target_label)

