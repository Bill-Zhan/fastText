# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:21:19 2018

@author: billzhan
"""

#==========   TEXT CLEANING   ==========#
import re
import pandas as pd



#%% GLobal
PATH = r"/home/billzhan/OneDrive/Academic/Umich/2018Winter/MDP_ProQuest Drug Safety Project/fastText/new_old/"
FILE = r"old_abstract1.csv"
STOPW_LIST = r"stopword_list"
NEWFILE = r"cleaned_oldabstract.csv"

#%% Functions
def read_data(path:str,file:str) -> 'pandas dataframe':
	'''
	read data into pandas data frame
	'''
	df = pd.read_csv(path+file)
	abstracts = df.abstract.tolist()

	return df,abstracts

def read_stopword(path,file) -> list:
    s = pd.read_table(path+file, header=None)
    stw_list = s[0].tolist()
    return stw_list

def rm_punctuation(abstracts:'list of long strings') -> list:
	'''
	1.
	remove puntuations and get all words into lower case
	'''
	nopunct_abstracts = []
	for abstract in abstracts:
		# turn into lower case
		all_lower = abstract.lower()
		# remove punctuation, substitute by whitespace
		no_punct = re.sub(r'[^\s\w-]',' ',all_lower)
		nopunct_abstracts.append(no_punct)

	return nopunct_abstracts

def chg_num(abstracts:'list of long strings') -> list:
	'''
	2.
	change numbers into special word NUM
    and cut by whitespace into wordlist
	'''
	numchged_abstracts = []
	for abstract in abstracts:
		numchged = re.sub(r'[0-9]+','NUM',abstract)
		wordlist = re.split(r'\s',numchged)
		numchged_abstracts.append(wordlist)

	return numchged_abstracts

def rm_stopword(abstracts:'list of lists',stopwlist:list) -> list:
	'''
	3.
	remove stopwords from pre-determined list
	'''
	nostopw_abstracts = []
	for abstract in abstracts:
		stopw_rmd = [word for word in abstract if not word in stopwlist]
		joined_rmd = ' '.join(stopw_rmd).strip()
		nostopw_abstracts.append(joined_rmd)

	return nostopw_abstracts

#%% Main
def main() -> 'clean text data and write to csv filr':
    df,abstracts = read_data(PATH,FILE)
    nopunct_abstracts = rm_punctuation(abstracts)
    numchged_abstracts = chg_num(nopunct_abstracts)
    stw_list = read_stopword(PATH,STOPW_LIST)
    npstopw_abstracts = rm_stopword(numchged_abstracts,stw_list)
    newdf = pd.DataFrame({'abstract':npstopw_abstracts,
                          'new_old':df['new_old'],
                          'serious':df['serious']})
    newdf.to_csv(PATH+NEWFILE,index=False)

if __name__ == '__main__':
	main()