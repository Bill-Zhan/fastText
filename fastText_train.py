# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:03:34 2018

@author: billj
"""
import fasttext
import sys

#=====  Global Variables  =====#

train_txt = sys.argv[1]  #training data
FTmodel_name = sys.argv[2]  #the name of model you want to build
Lprefix = "__label__"  #the label prefix used when stored textdata into txt files


#=====  main function  =====#

def main():
    '''
    will generate a file called "FTmodel_name.bin" in current directory
    '''
    classifier = fasttext.supervised(train_txt,FTmodel_name,label_prefix=Lprefix,
                                     epoch=100,word_ngrams=3,lr=0.25,ws=10,dim=300,
                                     loss='softmax',bucket=2000000)
    print("Done")
    
if __name__ == '__main__':
    main()