# coding: utf-8
from dataset import *
from torch.utils.data import DataLoader
from model import crnn
from decode import *
from one_hot import _hot,tokens
from torch.optim import SGD as sgd
from torch.nn.functional import ctc_loss
#ctc_loss(q,target.type(torch.int32),torch.Tensor([50]),torch.Tensor([17]))
words = telugu_data('train')
dataload= DataLoader(words, batch_size = 10,num_workers=4,shuffle=True)
model = crnn(88,2).cuda()
sgd = sgd(model.parameters(),lr=0.1,momentum=0.09)

tok_dict=dict(zip(tokens,np.arange(88)))
def _convert(char_string):
    t = []
    for i in char_string:
        t.append(tok_dict[i])
    return torch.Tensor(t).type(torch.int32)


def train():
    for i in range(len(words)):
        output = model(words[i][0].type(torch.cuda.FloatTensor)).log_softmax(2).detach().requires_grad_()
        target = _convert(words[i][1])
        loss = ctc_loss(output,target,torch.cuda.FloatTensor([50]),torch.cuda.FloatTensor([len(words[i][1])]))
        print(loss)
        sgd.zero_grad()
        loss.backward()
        sgd.step()
















    
if __name__ == '__main__':
    for i in range(10):
        train()

