import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import copy
from torch.autograd import Variable
from PIL import Image
import CIFAR_net
import os.path as osp
import numpy as np


def outlier_detection(l1_norm_list):
    l1_norm_list=np.clip(l1_norm_list,0.1,100)#set the minimal value
   # consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad =  np.median(np.abs(l1_norm_list - median))
    min_mad = 2*(np.max(l1_norm_list) - median) / (mad+median)
    return min_mad


def default_loader(path):
  #  image=cv2.imread(path)
  #  image = image[:,:,::-1]
   # image = image/255.0
   # image = image - 0.5
    image=Image.open(path).convert('RGB')
    return image#.astype(np.float)

class MyDataset(Dataset):
    def __init__(self,transform=None,target_transform=None, loader=default_loader):
         super(MyDataset,self).__init__()
         imgs=[]
         for i in range(10):
             for j in range(1000):
                  filename = str(j).zfill(4)+'.png'
                  words='./CIFAR10_dataset/test_image/'+str(i).zfill(1)+'/'+filename
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

def load_test_dataset(batch_size=80):
    train_transforms = transforms.Compose([
#        transforms.Resize(input_size),
#        transforms.RandomRotation(10),
#        transforms.CenterCrop(input_size),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
 #       transforms.Normalize((.5, .5, .5), (.5, .5, .5)) #to [-1,1]
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader

class MyDataset2(Dataset):
    def __init__(self,target_class=None,transform=None,target_transform=None, loader=default_loader):
         super(MyDataset2,self).__init__()
         imgs=[]

         for i in range(1):
             for j in range(1000):
                  filename = str(j).zfill(4)+'.png'
                  words='./CIFAR10_dataset/test_image/'+str(target_class).zfill(1)+'/'+filename
                  imgs.append((words,int(target_class)))

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

def load_target_dataset(batch_size=20,target_class=None):
    train_transforms = transforms.Compose([
#        transforms.Resize(input_size),
#        transforms.RandomRotation(10),
#        transforms.CenterCrop(input_size),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
 #       transforms.Normalize((.5, .5, .5), (.5, .5, .5)) #to [-1,1]
    ])

    train_datasets = MyDataset2( target_class, transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader


def add_AE(X,UAE,mask_tanh,alpha):
    X = X*(1-mask_tanh)+(UAE*alpha+(1-alpha)*X)*mask_tanh
    return X

def save_adv(UAE,path,target_label):
    combine_img=255*(UAE.cpu().data.numpy()[::-1,:,:].transpose(1,2,0))
    cv2.imwrite(path+'/star'+str(target_label)+'_trigger.png', combine_img)    

def save_mask(mask,path,target_label):
    combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
    cv2.imwrite(path+'/star'+str(target_label)+'_mask.png', combine_img)

def save_img(X,path):
    index=X.shape[0]
    for i in range(index):
        combine_img=255*(X[i,:,:,:].cpu().data.numpy()[::-1,:,:].transpose(1,2,0))
        #print(combine_img)
        cv2.imwrite(path+'/'+str(i)+'.png', combine_img)

def get_ave_logits(model):
    val_dataloader = load_test_dataset()
    n=0
    ave_logits=torch.zeros(10).cuda()
    ave_softlogits=torch.zeros(10).cuda()

    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        y_pred=model(X)

        y_pred_soft=torch.softmax(y_pred,1)
        for kk in range(len(y)):
           y_pred[kk,y[kk]]=0
           y_pred_soft[kk,y[kk]] = 0

        ave_logits=ave_logits+torch.sum(y_pred,dim=0)
        ave_softlogits=ave_softlogits+torch.sum(y_pred_soft,dim=0)
        n += y.shape[0]
    return ave_logits.data/(n-1000)



def nc_detect(test_loader2,model,target_label,clean_model):

    mask = torch.ones([1,32,32]).cuda()
    UAE = torch.ones([3,32,32]).cuda()
    alpha=torch.tensor(1)
    alpha=alpha.float().cuda()
    path='./trigger/'


    lossFN = nn.CrossEntropyLoss()
   # lossFN= nn.BCEWithLogitsLoss()
    epsilon = 0.03
    epsilon2=0.1
    epsilon3=0.05
    n=0
    sum_acc=0
    n_mask=0
    sum_mask=torch.zeros(10)
    n_accu=0
    sum_accu=torch.zeros(10)
    temp=9  #  cross_entropy 10
    for epoch in range(2):
        for i, (X, y) in enumerate(test_loader2):
            X=X.cuda().float()
            y=y.cuda()
            y_t=y.clone()
            y_t[:]=target_label
            
            UAE=Variable(UAE.data,requires_grad=True)
            mask=Variable(mask.data,requires_grad=True)
            alpha=Variable(alpha.data,requires_grad=True)
            mask_tanh=torch.tanh(8*(mask-0.5))/2+0.5
            noise=torch.randn(X.shape)*0.1
            X=X+noise.cuda()
            X_adv = add_AE(X.clone(), UAE, mask_tanh, alpha)
            #save_img(X_adv,'./test_img/')

            y_pred=model(X_adv)
           # print(y_pred.argmax(dim=1))
            sum_accu[n_accu]=(y_pred.argmax(dim=1) ==y_t).sum()/len(y_t)
            n_accu=n_accu+1
            ave_accu=torch.sum(sum_accu)/10
            if n_accu>9:
                n_accu=0
                if ave_accu<0.6:
                    temp=temp+10
                if ave_accu>0.9:
                    if temp>10:
                        temp=temp-10
                    else:
                        temp=temp/10

            loss_l1=torch.sum(torch.abs(mask_tanh))
            loss_pred=lossFN(y_pred,y_t)
            loss=loss_pred+loss_l1/temp
            loss.backward()
            AE_grad=UAE.grad
            Mask_grad=mask.grad
            Mask_grad=torch.sum(Mask_grad,dim=0)

            perturb =epsilon*torch.sign(AE_grad)
            UAE=UAE-perturb
            UAE_mean=torch.sum(UAE,dim=0)/3
            UAE_mean=UAE_mean.unsqueeze(0)
            UAE_mean=UAE_mean.repeat(3,1,1)
            mx=(UAE>UAE_mean).float()
            mi=1-mx
            UAE_mx=torch.min(UAE,UAE_mean+0.1)*mx
            UAE_mi=torch.max(UAE,UAE_mean-0.1)*mi
            UAE=UAE_mx+UAE_mi

            UAE=torch.clamp(UAE,0,1)

            perturb2 =epsilon2*torch.sign(Mask_grad)
            mask=mask-perturb2
            mask=torch.clamp(mask,0,1)
            sum_mask[n_mask]=torch.sum(mask)
            n_mask=n_mask+1
            if n_mask>9:
               n_mask=0
            ave_mask=torch.sum(sum_mask)/10
            n=n+1
            if n>1000:
                break
   # save_adv(UAE,path,target_label)
   # save_mask(mask_tanh,path,target_label)
    y_pred_clean=model_clean(X_adv)
  #  print(temp)
 #   print(torch.sum(UAE))
    accu_clean=(y_pred_clean.argmax(dim=1) ==y_t).sum()/len(y_t)

    return ave_accu,ave_mask,accu_clean

def model_detect(test_loader,target_loader_k,model,ori_model,k,model_clean,lk):
    features=list(model.parameters())
    para1=[features[i+8]  for i in range(8)]
    optimizer = torch.optim.SGD(para1, lr=2e-3, momentum=0.9, weight_decay=5e-4)
    lossFN = nn.CrossEntropyLoss()
    loss2_sum=torch.zeros(10)
    loss2_n=0
    for epoch in range(5):

       target_iterator=iter(target_loader_k)
       for i, (X, y) in enumerate(test_loader):
                if i%50==0:
                    target_iterator=iter(target_loader_k)
                X=X.float().cuda()
                y=y.cuda()
                X_t,y_t= next(target_iterator)
                X_t=X_t.float().cuda()
                y_t=y_t.cuda()

                X_in=torch.cat([X,X_t], dim=0)
                #save_img(X_in,'./test_img/')
                y_pred=model(X_in)
                loss1=lossFN(y_pred[0:len(y)],y)
                loss2=0
                y_p=torch.softmax(y_pred, dim=1)
                for t in range(len(y_t)):
                    loss2=loss2+y_p[len(y)+t,y_t[t]]

                with torch.no_grad():
                    out1 = ori_model(X)
                logit_sim_loss = (out1.detach() - y_pred[0:len(y)]).view(X.shape[0], -1).norm(p=1, dim=1).mean(0)
                logit_loss=(torch.abs(y_pred[len(y)-1:-1,k]-lk[k])).mean(0)
                loss=loss1+logit_loss+logit_sim_loss/10
                loss.backward()
                loss2_sum[loss2_n]=loss2
                loss2_n=loss2_n+1
                if loss2_n>9:
                    loss2_n=0
                loss2_ave=torch.sum(loss2_sum)/10

                optimizer.step()
                optimizer.zero_grad()
                if loss2_ave<1:
                    break
       if loss2_ave<1:
            break
       n=0


    accu,mask,accu_clean=nc_detect(test_loader,model,k,model_clean)
    return accu,mask,accu_clean


def single_model_detect(test_loader,target_loader,model_path,model_clean):
   ratio=0.2
   #load original model
   ori_model =  CIFAR_net.model_def(True,model_url=model_path).cuda()
   ori_model.cuda().eval()
   lk=get_ave_logits(ori_model)
   
   for kk in range(1):
       result=[]
       ratio=[]
       for k in range(10):
           model = copy.deepcopy(ori_model)
           accu,mask,accu_clean=model_detect(test_loader,target_loader[k],model,ori_model,k,model_clean,lk)
           result.append([accu,mask,accu_clean])
           ratio.append(100*accu/mask.data)
           #print(k,accu,mask,100*accu/mask,'transferability:',accu_clean)
           print(k,'acc/mask:', (100 * accu / mask).data, 'transferability:', accu_clean.data)
       if max(ratio)>1:
            break
   return result,ratio

if __name__ == "__main__":
   test_loader = load_test_dataset(batch_size=80)

   target_loader=[]
   for tk in range(10):
       target_loader_k=load_target_dataset(batch_size=20,target_class=tk)
       target_loader.append(target_loader_k)

   modelclean_path='./clean_model/model_clean.pth'
   model_clean = CIFAR_net.model_def(True,model_url=modelclean_path).cuda()
   model_clean.cuda().eval()
   models_path='./model_to_detect'

   try:
       imlist = [osp.join(osp.realpath('.'), models_path, model) for model in os.listdir(models_path) if
                 os.path.splitext(model)[1] == '.pth']
   except FileNotFoundError:
       print("No file or directory")
       exit()


   for ml in range(len(imlist)):
       print('model:',imlist[ml])
      # model_path="../backdoor_model_file/model_"+str(model_num).zfill(2)+".pth"
       result,ratio=single_model_detect(test_loader,target_loader,imlist[ml],model_clean)
       anomaly_index = outlier_detection(ratio)
       print('anomaly_index:',anomaly_index)





