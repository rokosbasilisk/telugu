
# coding: utf-8
from model import crnn
import torch
from skimage.io import imread 
from skimage.transform import resize
from torch.autograd import Variable
from decode import beam_search
import numpy as np
import sys
from torch.autograd import Variable
from dataset import im2s
model = crnn(88,2)
model.load_state_dict(torch.load('./model40000.pth'))

def test_image(image_path):
    outs=im2s(imread(image_path),lang='tel')
    print(outs.split('\n'))

def _test_image(image_path):
    img = imread(image_path)
    img = resize(img,(224,224))
    img = torch.Tensor(np.rollaxis(img,2,0))
    outs=model(Variable(img).unsqueeze(0))
    outs = np.rollaxis(outs.detach().numpy(),2,1)
    outs=np.rollaxis(outs,1,2).reshape(88,50)
    outs=beam_search(outs)
    print(outs)

"""
im = imread('./1.jpg')
im = resize(im,(224,224))

im=np.rollaxis(im,2,0)
im
model(im)
model(torch.Tensor(im))
im = Variable(torch.Tensor(im)).unsqueeze(0)

im = Variable(torch.Tensor(im)).unsqueeze(0)
im
model(im)
outs=model(im)
out.shape
outs.shape

beam_search(outs)
beam_search(outs.detach().numpy())
outs.shape
out=outs.reshape(88,50)
beam_search(outs.detach().numpy())
outs=model(im)
beaim_search(out.detach().numpy())
"""
if __name__=='__main__':
    test_image(sys.argv[1])
    #_test_image(sys.argv[1])
