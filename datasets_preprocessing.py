#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Text_Cnn import word2index
import nltk

def clean_datasets(docs):
    """
    clean data for Word2vec Emnbedding
    """
    punct = [',','.',':','(',')','?','!','-']
    preposition = ['to','of','and','a']
    remove_list = punct + preposition
    for docid in docs:
        doc = docs[docid]
        #remove words
        doc = list(filter(lambda x: x not in remove_list, doc))
        #replace words
        for i,word in enumerate(doc):
            if word == "'s":
                doc[i] = 'is'
            elif word == "n't":
                doc[i] = 'not'
            elif word == "'re":
                doc[i] = 'are'
        #return cleaned doc    
        docs[docid] = doc
    return docs

# raw data -> dataset
# some necessary function


def get_label(dataset):
    return dataset.classification
'''
def get_docid(dataset):
    return c.annotation_id
'''
# create the dataset class
class Movie_Classif_Dataset(Dataset):
    """
    create the dataset class for Movie sentiment classification
    """
    def __init__(self, docs, annos):
        # run once when init
        self.docs = docs
        self.annos = annos
        self.labels = list(map(get_label,annos))
        
    def __len__(self):
        #returns the number of samples in our dataset
        return len(self.labels)
    
    def __getitem__(self, idx):
        #returns a sample from the dataset at the given index
        label = self.labels[idx]
        docid = self.annos[idx].annotation_id
        text = self.docs[docid]
        words_id = word2index(text)
        #sample = {"Text": text, "Class": label}
        return words_id,label

def get_part_of_speech(w):
    '''
    get the part of speech of a single word
    input: word
    output:part of speech
    '''
    word = []
    word.append(w)
    word_tag = nltk.pos_tag(word)
    return word_tag[0][1]

def update_tag_dict(sentence,old_dict):
    '''
    update word_tag_dict
    input: sentence, old_dict
    output: updated word_tag_dict
    '''
    word_tag_dict = old_dict
    word_tag = nltk.pos_tag(sentence)
    for (word,tag) in word_tag:
        if word in word_tag_dict:
            word_tag_dict[word].add(tag)
        else:
            word_tag_dict[word] = set()
            word_tag_dict[word].add(tag)
        
    return word_tag_dict

def clean_sentence(sentence):
    '''
    clean sentence
    input: sentence
    output: cleaned sentence
    '''
    punct = [',','.',':','(',')','?','!','-']
    preposition = ['to','of','and','a']
    remove_list = punct + preposition
    #remove words
    sent = list(filter(lambda x: x not in remove_list, sentence))
    #replace words
    for i,word in enumerate(sent):
        if word == "'s":
            sent[i] = 'is'
        elif word == "n't":
            sent[i] = 'not'
        elif word == "'re":
            sent[i] = 'are'
    return sent

def count_tag_freq(documents):
    '''
    A function to count part of speech's tag frequence 
    input: documents
    output: tag_freq
    '''
    # init tags_count and words_num
    tag_list = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG',
            'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', 
            ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.',
            ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 
            'FW','RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 
            'NNP','EX', 'NNS', 'SYM', 'CC', 'CD', 'POS','#']
    tags_count = {}
    for tag in tag_list:
        tags_count[tag] = 0
    words_num = 0
    # count tags nummer and words nummer
    for doc_id in documents:
        print(doc_id)
        doc = documents[doc_id]
        for sentence in doc:
            sentence = clean_sentence(sentence)
            words_tag = nltk.pos_tag(sentence)
            for (word,tag) in words_tag:
                tags_count[tag] += 1
                words_num += 1
    
    return words_num, tags_count

def doc_to_tag(doc):
    '''
    get a tag list for each word in a sentence_level_doc
    input: sentence_level_doc
    output: tag for each word in doc
    '''
    tag_list = []
    for sentence in doc:
        sentence = clean_sentence(sentence)
        word_tag = nltk.pos_tag(sentence)
        for (word,tag) in word_tag:
            tag_list.append(tag)
    return tag_list

def get_word_tag_dict(documents):
    '''
    get possible tag for words in this documents
    input: sentence level documents
    output: a dictionary, key is word, value is word's tag
    '''
    word_tag_dict = {}
    for doc_id in documents:
        doc = documents[doc_id]
        for sentence in doc:
            # clean the sentence
            sentence = clean_sentence(sentence)
            # update word_tag_dict
            word_tag_dict = update_tag_dict(sentence,word_tag_dict)
            
    return word_tag_dict

def get_wi_tag(sent_level_doc,word_place,wi):
    '''
    input: sententce level document, w0_place in document, wi
    ruturn: tag of wi after replace w0
    '''
    # return the sentence according to the word position,and change w0 to wi
    current_position = 0
    for sentence in sent_level_doc:
        sent = clean_sentence(sentence)
        l = len(sent)
        current_position += l
        if current_position > word_place:
            w0_position_in_sent = word_place-(current_position-l)
            sent[w0_position_in_sent] = wi
            break
    
    # get wi_tag
    tags = nltk.pos_tag(sent)
    return tags[w0_position_in_sent][1]