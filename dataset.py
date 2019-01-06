import torch
from torch.utils.data import Dataset
import os
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt
import sys
from skimage.transform import resize
import torch
from torch.autograd import Variable
import numpy as np
class telugu_data(Dataset):

    def __init__(self,subset):
        self.subset=subset
        with open('./data/'+subset+'.txt') as f:
            img_lis=f.read()
            img_lis=img_lis.split('\n')
        self.img_lis=img_lis

    def __len__(self):
        return len(self.img_lis)
    def __getitem__(self,i):
        img=imread('./data/'+self.img_lis[int(i)].split()[0])
        img=resize(img,(224,224))
        img_array=img
        lbl=self.img_lis[int(i)].split()[1]
        img = np.rollaxis(img,2,0)
        img=Variable(torch.Tensor(img)).unsqueeze(0)
        return(img,lbl,img_array,len(lbl))
    def get_batch(self,indices):
        image_batch=[]
        label_batch=[]
        for i in indices:
            img = imread('./data/'+self.img_lis[int(i)].split()[0])
            img=resize(img,(224,224))
            lbl=self.img_lis[int(i)].split()[1]
            img=np.rollaxis(img,2,0)
            img=Variable(torch.Tensor(img)).unsqueeze(0)
            image_batch.append(img.view(3,224,224))
            label_batch.append(lbl)
        image_batch=Variable(torch.stack(image_batch))
        return image_batch,label_batch

