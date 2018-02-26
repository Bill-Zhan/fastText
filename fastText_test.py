# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:52:27 2018

@author: billj
"""
#Note: this can only be run in the Linux environment

import fasttext
import sys

#=====  Global Variables  =====#
testdata = sys.argv[1]
modelname = sys.argv[2]
Lprefix = '__label__'


#===== main function  =====#

def main(): 
    classifier = fasttext.load_model(modelname, label_prefix=Lprefix)
    result = classifier.test(testdata)
    print (result.precision,result.recall)

if __name__=='__main__':
    main()