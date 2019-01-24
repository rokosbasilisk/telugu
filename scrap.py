# coding: utf-8
from dataset import telugu_data
import torch.utils.data.dataloader as dataloader
dataloader
get_ipython().run_line_magic('pinfo', 'dataloader.DataLoader')
q
words = telugu_data('train')
dl=dataloader.DataLoader(words,batch_size=10)
dl
