# coding: utf-8
import os
import numpy as np

tokens = open('./data/letters.txt','r').read()
tokens=tokens.split()

tokens=list(tokens[0])
tokens.append('')

tokens=sorted(tokens)
    
dic=dict(zip(tokens,np.arange(88)))

def one_hot(char):
    index = np.zeros(88)
    index[dic[char]]=1
    return index
def _hot(word):
    l=[]
    for i in word:
        l.append(one_hot(i))
    return l


    

if __name__ == '__main__':
    for i in range(5):
        print(one_hot(tokens[i]))
