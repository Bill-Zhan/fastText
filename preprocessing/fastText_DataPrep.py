# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:13:34 2018

@author: billj
"""
import pandas as pd
import numpy as np
import re
import os
#=====  Global Variables  =====#

#--- directory and file name
wkdir = r'C:\Users\billj\OneDrive\Academic\Umich\2018Winter\MDP_ProQuest Drug Safety Project\fastText'
raw_data = r'serious_data.csv'

#--- map original long class names to some short dummy labels
dummy_map = {'Death / Life Threatening':'D','Serious':'S','Non-Serious':'NS'}

#--- set up train and test txt files which fastText uses as inputs
traintxt_name = r"serious_FT_train.txt"
testtxt_name = r"serious_FT_test.txt"


#=====  Functions  =====#

#--- load data
def loaddata(csvfile:'with 2 cols: text and label')->list:
    '''
    csvfile -- a csv file restore texts and relevant labels
    '''
    data = pd.read_csv(csvfile)
    text = data.text.tolist()
    label = data.serious.tolist()
    textdata = []
    for line in text:
        doc = re.sub(r'\[|\]|\'|\(|\)|\'|\*|\:','',line)
        d = doc.split(',')
        clean = [word.strip() for word in d]
        textdata.append(clean)
    return textdata,label

#--- create dummmy labels
def create_dummy(labels:list,dummy_map:dict) -> list:
    '''
    create fastText desired labels when original class names are too long
    '''
    dummy_label = []
    for i in range(len(labels)):
        dummy_label.append(dummy_map[labels[i]])
    return dummy_label

#--- partition data
def part_data(feature:"textdata",label:list,p=0.8)->list:
    n = len(label)
    feature = np.array(feature)
    label = np.array(label)
    train_ix = np.random.choice(np.arange(n),size=int(0.8*n),replace=False)
    test_ix = np.setdiff1d(np.arange(n),train_ix)
    train_x,train_y = feature[train_ix],label[train_ix]
    test_x,test_y = feature[test_ix],label[test_ix]
    return list(train_x),list(train_y),list(test_x),list(test_y)


#=====  main function  =====#
def main():
    #---import data

    #read data
    os.chdir(wkdir)#change to current working directory
    textdata,label = loaddata(raw_data)
    '''
    If you have well-segmented text file, that is,
    each line consists of the segmented words of an observation,
    you can directly use this:
    '''
    #textdata = your_text_file
    
    #---create dummy label in case the classnames to long
    dummy_label = create_dummy(label,dummy_map)
    
    #---create train and test datasets
    train_x,train_y,test_x,test_y = part_data(textdata,dummy_label)
    
    #---store in format that fastText needs

    #open txt files to write in data
    ftrain = open(traintxt_name,"wb")
    ftest = open(testtxt_name,"wb")
    #write train txt file
    for i in range(len(train_x)):
        textline = ' '.join(train_x[i])
        output = textline.encode("utf-8") + b"\t__label__" + train_y[i].encode("utf-8") + b"\n"
        outline = output
        ftrain.write(outline)
        ftrain.flush()
    #write test txt file
    for i in range(len(test_x)):
        textline = ' '.join(test_x[i])
        output = textline.encode("utf-8") + b"\t__label__" + test_y[i].encode("utf-8") + b"\n"
        outline = output
        ftest.write(outline)
        ftest.flush()
    ftrain.close()
    ftest.close()

if __name__=='__main__':
    main()
    
    
    





















