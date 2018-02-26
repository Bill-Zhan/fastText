# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:27:57 2018

@author: billj
"""

import re
import os
import pandas as pd
from bs4 import BeautifulSoup

#===== Global  =====#

WKDIR = r"C:\Users\billj\OneDrive\Academic\Umich\2018Winter\MDP_ProQuest Drug Safety Project"
FILENAME = r"data_source\reportable.csv"

#--- map original long class names to some short dummy labels
dummy_map = {'Death / Life Threatening':'D','Serious':'S','Non-Serious':'NS'}

#--- set up train and test txt files which fastText uses as inputs
word2vec_name = r"fastText\abstract2vec.txt"


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
def clean_tag(indexing:'pd.dataframe')->list:
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
def alphabet_text(notag_text)->list:
    clean_text = []
    for row in notag_text:
        no_PuncNum = re.sub(r'[^\s\w-]|[0-9]','',row)
        clean_text.append(no_PuncNum)
    #---split strings and remove space
    splitted_text = [row.split(' ') for row in clean_text]
    #---change all to lower case
    final_text = []
    for row in splitted_text:
        final_text.append(list(map(lambda word:word.lower(),row)))
    return final_text

#======  Main  =====#
def main():
    #---prepare data
    text_toclean = report_abstract.author_abstract
    notag_text = clean_tag(text_toclean)
    textdata = alphabet_text(notag_text)
    
    #---store in format that fastText needs

    #open txt files to write in data
    f2vec = open(word2vec_name,"wb")

    #write train txt file
    for i in range(len(textdata)):
        textline = ' '.join(textdata[i])
        output = textline.encode("utf-8") + b"\n"
        outline = output
        f2vec.write(outline)
        f2vec.flush()

    f2vec.close()
    
if __name__=="__main__":
    main()













