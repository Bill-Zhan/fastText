# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:43:13 2018

@author: billj

*** supplementary module in billnlp ***

"""
import re
import numpy as np
from collections import Counter
from functools import reduce


#%% Functions
def word_doc(file)->'number of docs, a word:num docs dict':
    with open(file) as f:
        alldocs = f.readlines()
        ndocs = len(alldocs)
        alldocs = [line.strip('\n') for line in alldocs]  #remove \n
        alldocs = [line.split(' ') for line in alldocs]  #remove space
        allwords = list(set(reduce(lambda x,y:x+y, alldocs)))
        word_docs = {}
        for word in allwords:
            indocs = [1 if word in line else 0 for line in alldocs]
            word_docs[word]=sum(indocs)
        
        return ndocs,word_docs
        
        
def Tf_Idf(file:'txt file with each line a document',
           ndocs:int,wd_dict:'number of docs of each word')->'list of dicts with weight in each one':
    with open(file) as f:
        # initialization
        tfidf_result = []
        for line in f:
            # glovar vars for each doc
            nwords = len(line)
            line = line.strip('\n')
            wordlist = line.split(' ')
            # compute tf
            word_counts = dict(Counter(wordlist))
            tf = {word: count/nwords for word,count in word_counts.items()}
            # compute idf
            idf = {word: np.log(ndocs/(wd_dict[word]+1)) for word in word_counts.keys()}
            # tf-idf is the product of tf and idf
            tf_idf = {word:tf[word]*idf[word] for word in word_counts.keys()}  #create a dictionary for each document
            tfidf_result.append(tf_idf)
    
    return tfidf_result
            



































