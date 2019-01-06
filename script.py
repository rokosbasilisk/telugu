# coding: utf-8
from dataset import telugu_data
from model import crnn
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow
import torch.nn as n
import torch.nn as nn
from nn.functional import ctc_loss
from torch.nn.functional import ctc_loss
import torch
from one_hot import tokens
model = crnn(88,2)
words = telugu_data('train')

len(tokens)
tokens[-1]
tok_dict=dict(zip(tokens,np.arange(88)))

tok_dict=dict(zip(tokens,np.arange(88)))

ctc_loss(q,target.type(torch.int32),torch.Tensor([50]),torch.Tensor([17]))

def _convert(char_string):
    t = []
    for i in char_string:
        t.append(tok_dict[i])
    return torch.Tensor(t).type(torch.int32)

