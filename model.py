import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet18 as resnet

resnet = resnet(pretrained=False)

model = nn.Sequential()
layer_number=0
for layer in list(resnet.children())[:-1]:
    layer_number=layer_number+1
    model.add_module('layer  %d'%layer_number,layer)

del(resnet)

class _rnn(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(_rnn, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class crnn(nn.Module):

    def __init__(self, nclass, n_hidden):
        super(crnn, self).__init__()
   
        self.cnn = model
        self.rnn = nn.Sequential(_rnn(512, n_hidden, n_hidden),_rnn(n_hidden, n_hidden, nclass))
        self.fc=nn.Linear(88,4400)
        self.softmax=nn.Softmax(dim=2)
    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        b=conv.shape[1]
        # rnn features
        output = self.rnn(conv)
       #print(conv.shape) 
        output=self.fc(output)
        output=output.view(88,b,50)
        output=self.softmax(output)
       #print(output.shape)
        return output

#w,b,c are sizes given to the rnn layer width,batchsize,channels? since from convnet


if __name__=='__main__':
    model = crnn(88,2)
    model.children()
   #print(list(model.children()))
    from dataset import telugu_data
    words = telugu_data('train')
    q=words.get_batch([1,2,3,4])
    q=model(q[0])
    print(q)
    print(q.shape)




