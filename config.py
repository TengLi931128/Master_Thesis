'''
This is a config file of Text_Cnn
'''
import numpy as np
import torch

class DefaultConfig(object):
    # Data_set
    data_root = ''
    train_data_root = ''
    test_data_root = ''
    eva_data_root = ''
    
    # Word2Vec model
    vocab_size = 300000
    word_size = 300
    unk_vec = np.zeros(word_size,dtype=np.float32) #unknown vector
    pad_vec = np.zeros(word_size,dtype=np.float32) #padding vector
    
    # Text_CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available
    filter_size = [3,4,5]        #filter size of CNN
    filter_num = [100,100,100]   #number of each filter size
    
    # BI-LSTM 
    lstm_batch_size = 2
    input_size = word_size # input size of lstm
    hidden_size =50        # hidden size
    num_layers = 2         # 2 layers lstm
    bidirectional = 2   # Bidirectional
    batch_first = True
    num_class = 2
    
    # Training
    lr = 0.05           
    epochs = 30        
    batch_size = 2     
    print_freq = 100   # every 100 batchs print info
    
    # Hotflip
    beam_search_size = 10 # how many words saved to campare
    change_word_num = 5 # how many words should flip for each text
    keep_word_num = 3   # how many words should be keep for each fliped word
    search_size = 2000
    


