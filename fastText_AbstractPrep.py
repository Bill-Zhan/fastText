# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:01:19 2018

@author: billj
"""
import re
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

#===== Global  =====#

WKDIR = r"C:\Users\billj\OneDrive\Academic\Umich\2018Winter\MDP_ProQuest Drug Safety Project"
FILENAME = r"data_source\reportable.csv"

#--- map original long class names to some short dummy labels
dummy_map = {'Death / Life Threatening':'D','Serious':'S','Non-Serious':'NS'}

#--- set up train and test txt files which fastText uses as inputs
traintxt_name = r"fastText\abstract_train.txt"
testtxt_name = r"fastText\abstract_test.txt"


#---read data
os.chdir(WKDIR)
reportable_data = pd.read_csv(FILENAME,encoding='cp1252')
reportable_abstract = reportable_data[reportable_data["author_abstract"].notna()]

#---select data to process
seriousness = ['Death / Life Threatening','Serious','Non-Serious']
df = reportable_abstract[['author_abstract','icsr_assesment']]
report_abstract = df[df['icsr_assesment'].isin(seriousness)]


#=====  Functions  =====#

#--- remove tags
def clean_tag(indexing:'pd.dataframe'):
    '''
    remove tag info and numbers in the indexing variable
    '''
    clean_text = []
    textdata = indexing.values
    for row in textdata:
        soup = BeautifulSoup(row)
        text = soup.get_text()
        clean_text.append(text)
    
    return clean_text

#--- remove non alphabet
def alphabet_text(notag_text):
    clean_text = []
    for row in notag_text:
        no_PuncNum = re.sub(r'[^\s\w-]|[0-9]','',row)
        clean_text.append(no_PuncNum)
    #--- split strings and remove space
    splitted_text = [row.split(' ') for row in clean_text]
    return splitted_text

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
     

#=====  main  =====#


def main():
    #---prepare data
    text_toclean = report_abstract.author_abstract
    notag_text = clean_tag(text_toclean)
    textdata = alphabet_text(notag_text)
    label = report_abstract['icsr_assesment'].tolist()
    
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





















