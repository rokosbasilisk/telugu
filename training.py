# coding: utf-8
from dataset import *
from torch.utils.data import DataLoader
from model import crnn
from decode import *
from one_hot import _hot,tokens
from torch.optim import SGD as sgd
from torch.nn.functional import ctc_loss
import torch
from tqdm import tqdm
#ctc_loss(q,target.type(torch.int32),torch.Tensor([50]),torch.Tensor([17]))
words = telugu_data('train')
dataload= DataLoader(words, batch_size = 10,num_workers=4,shuffle=True)
model = crnn(88,2).cuda()
sgd = sgd(model.parameters(),lr=0.01,momentum=0.9)

tok_dict=dict(zip(tokens,np.arange(88)))
def _convert(char_string):
    t = []
    for i in char_string:
        t.append(tok_dict[i])
    return torch.Tensor(t).type(torch.int32).cuda()


def train():
    count=0
    for i in tqdm(range(len(words))):
        count=count+1
        output = model(words[i][0].type(torch.cuda.FloatTensor)).log_softmax(2).detach().requires_grad_()
        target = _convert(words[i][1])
        loss = ctc_loss(output,target,torch.cuda.FloatTensor([50]),torch.cuda.FloatTensor([len(words[i][1])]))
        if(count%50==0):
            print(loss)
        if(count%10000==0):
            torch.save(model.state_dict(), './model%d.pth'%count)
        sgd.zero_grad()
        loss.backward()
        sgd.step()
"""
def batch_train(epochs):
    for epoch in range(epochs):
        for i,batch in enumerate(dataload):

            data_batch=torch.Tensor(batch[0]).view(10,3,224,224)
            data_batch=data_batch
            output = model(data_batch)
            target_words=[]
            target_lengths=[]
            for k in batch[1]:
                target_words.append(torch.Tensor(_convert(k)))
                target_lengths.append(len(k))

            ###error    
            target_words = torch.Tensor(target_words)
            target_lengths=torch.Tensor(target_lengths)
            input_lengths=torch.FloatTensor(50*np.ones(10))
            
            ###trainingloop
            loss = ctc_loss(output,target_words,input_lengths,target_lengths)
            print(ss)
            sgd.zero_grad()
            loss.backward()
            sgd.step()
    return
"""
if __name__ == '__main__':
    for i in tqdm(range(10)):
        train()

















    
