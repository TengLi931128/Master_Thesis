#!/usr/bin/env python
# coding: utf-8



'''
this is a Bi_Lstm model, which will be used for text classification
Teng Li
06,Dec,2021
'''

from config import DefaultConfig
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as f

# In[3]:


# first of all get config
Conf = DefaultConfig()

DEVICE = Conf.device
LR = Conf.lr
BATCH_SIZE = 3 # get from Conf.lstm_batch_size
INPUT_SIZE = 300 # get from Conf.input_size
HIDDEN_SIZE = 50 # get from Conf.hidden_size
NUM_LAYERS = 2   # get from Conf.num_layers
BIDIRECTIONAL = 2   #get from Conf.bidirectional
NUM_CLASS = 2    # get from Conf.num_class

# then we can load word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('/home/teng/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz',
                                                        binary=True,limit=300000)
# first add 'unk' and 'pad'to our model
model['unk'] = Conf.unk_vec
model['pad'] = Conf.pad_vec
# the weights of Embedding layer are
W2V_weights = torch.FloatTensor(model.vectors)
# the vocab of this Embedding is
Vocab = model.wv.vocab
# get index of 'unk' and 'pad'
Unk_index = Vocab['unk'].index
Pad_index = Vocab['pad'].index


class Text_Bi_LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        #Embedding layer
        self.embed = nn.Embedding.from_pretrained(W2V_weights)
        self.embed.weight.requires_grad = True
        #LSTM layers (2 layers bi_lstm)
        self.lstm = nn.LSTM(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,
                            bidirectional=(BIDIRECTIONAL==2),batch_first=True)
        # init h0,c0
        self.h0 = nn.Parameter(torch.zeros(NUM_LAYERS*BIDIRECTIONAL,BATCH_SIZE,HIDDEN_SIZE))
        self.c0 = nn.Parameter(torch.zeros(NUM_LAYERS*BIDIRECTIONAL,BATCH_SIZE,HIDDEN_SIZE))
        # fc layer
        self.fc1 = nn.Linear(HIDDEN_SIZE*NUM_LAYERS*BIDIRECTIONAL,20)
        self.fc2 = nn.Linear(20,NUM_CLASS)
    
    def forward(self,x):
        # get batch_size
        batch_size = x.size(0)
        # reshape h0 and c0
        #h0 = self.h0[:,:1,:]
        #c0 = self.c0[:,:1,:]
        # embeding layer
        x = self.embed(x)
        # lstm layers
        out,(hn,cn) = self.lstm(x,(self.h0,self.c0))
                
        x = hn           #(Num_layers*Bidirectional,Bat_size,hidden_size)
        x = x.permute(1,0,2) #(Bat_size,Num_layers*Bidirectional,hidden_size)
        x = x.reshape(batch_size,-1)
              
        # fc layers
        x = self.fc1(x)
        x = self.fc2(x)
        # log_softmax
        x = f.log_softmax(x,dim=1)
        return x
    
    def get_word_embedding(self,word_id):
        embedding = self.embed(word_id)
        return embedding
    
    def get_gradient(self,x,y,loss_func=nn.NLLLoss()):
        # get batch_size
        batch_size = x.size(0)
        # embeding layer
        embed = self.embed(x)
        # lstm layers
        out,(hn,cn) = self.lstm(embed,(self.h0,self.c0))
                
        x = hn           #(Num_layers*Bidirectional,Bat_size,hidden_size)
        x = x.permute(1,0,2) #(Bat_size,Num_layers*Bidirectional,hidden_size)
        x = x.reshape(batch_size,-1)
              
        # fc layers
        x = self.fc1(x)
        x = self.fc2(x)
        # log_softmax
        y_hat = f.log_softmax(x,dim=1)
        # then comput loss
        loss = loss_func(y_hat,y)
        # get gradient
        gradient = torch.autograd.grad(loss,embed)
        return gradient
    



def lstm_padding(Dataset):
    '''
    padding for each batch
    '''
    len_max = 0
    #get max length
    for d in Dataset:
        l = len(d[0][0])
        if l>len_max:
            len_max = l
    #padding
    texts_id = []
    labels = []
    for d in Dataset:
        words_id = d[0][0][:len_max].tolist()
        label = d[1]
        if label == 'POS':
            label = 0
        else:
            label = 1
        labels.append(label)
        l = len(words_id)
        if l<len_max:
            padding = [Pad_index for _ in range(len_max-l)]
            words_id.extend(padding)
        texts_id.append(words_id)
    #list-> tensor
    labels = torch.tensor(labels)
    texts_id = torch.tensor(texts_id,dtype=torch.long)
    return texts_id,labels


