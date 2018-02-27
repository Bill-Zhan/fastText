# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:10:37 2018

@author: billj

*** supplementary module in billclassify ***

"""
import numpy as np

#%%  Functions

#---Uniformly partition data
def unif_partition(feature:list,label:list,
                   prop:'proportion of training data'=0.8)->'four lists':
    """
    try to partition the whole data set into training set and testing set according to classes
    i.e. sample the same proportion in each class to get training set
    in order to deal with inbalance data
    """
    # initialization
    n = len(label)
    X = np.array(feature)  #n*p matrix
    Y = np.array(label)  #n*1 vector
    class_names = list(set(label))
    train_index = np.array([],dtype=int)
    
    # for loop to trainset index
    for oneclass in class_names:
        ix = np.where(Y==oneclass)[0]  #np.where will return a tuple, whose 1st ele is index
        nset = len(ix)
        
        train_ix = np.random.choice(ix,size=int(prop*nset),replace=False)  #choose nset*prop samples from subset index
        train_index = np.append(train_index,train_ix)

    # get testset index
    test_index = np.setdiff1d(np.arange(n),train_index)
    
    # partition dataset
    train_x,train_y = X[train_index],Y[train_index]
    test_x,test_y = X[test_index],Y[test_index]
    return list(train_x),list(train_y),list(test_x),list(test_y)
































