# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:03:34 2018

@author: billj
"""
import fasttext
import sys
import argparse

#%%  Global Variables

Lprefix = "__label__"  #the label prefix used when stored textdata into txt files


#%% Main

def main():
	#--- create arg parser
	parser = argparse.ArgumentParser()

	#--- add arguments
	parser.add_argument('-i', type=str, help='input file name')
	parser.add_argument('-m', type=str, help='model name')
	parser.add_argument('-wng', type=int, choices=[1,2,3],help='number of word ngrams')
	parser.add_argument('-ws', type=int, help='window size')
	parser.add_argument('-dm', type=int, help='dimension of vector')
	
	#--- create args 
	args = parser.parse_args()

	#--- get variables
	if args.i:
		inputfile = args.i
	if args.m:
		modelname = args.m
	if args.wng:
		ngrams = args.wng
	if args.ws:
		windowsize = args.ws
	if args.dm:
		dimension = args.dm
	
	classifier = fasttext.supervised(inputfile,modelname,label_prefix=Lprefix,
    	epoch=100,word_ngrams=ngrams,lr=0.25,ws=windowsize,dim=dimension,
    	loss='softmax',bucket=2000000)
	
	print("Done")

if __name__ == "__main__":
    main()
