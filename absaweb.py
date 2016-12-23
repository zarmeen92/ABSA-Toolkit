# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:12:48 2016

@author: Zarmeen
"""

import warnings
warnings.filterwarnings("ignore")
import gensim
from gensim.models import Word2Vec
import sys
sys.path.insert(0, sys.path[0]+'\\flaskWebApp')
import flaskDemo
def main():
		print sys.argv
		# sys.argv[1] -- > 'vectors_yelp_200.txt'
		# sys.argv[2] -- > 'lexicons/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt'
		
		# python absaweb.py vectors_yelp_200.txt lexicons/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt
		# python absaweb.py amazon200.txt lexicons/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt
		
		
		vectors_filename = "wordembeddings/" + str(sys.argv[1]) #user provided
		lex_file =  str(sys.argv[2])
		while(True):
			flaskDemo.runMain(vectors_filename,lex_file)
		
main()
    
    
    	
    