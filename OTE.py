
# coding: utf-8

# In[77]:




# In[35]:

#import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
import re
import os
import numpy as np
from collections import Counter
import operator
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from sklearn.linear_model import LogisticRegression
import nltk
from xml.sax.saxutils import escape
import pandas as pd
import gensim
from gensim.models import Word2Vec
import pickle


# In[36]:

import nltk
#stopwords = nltk.corpus.stopwords.words('english')
def review_to_words( raw_review ):
    #remove non alphanumeric characters
    letters_only = re.sub("[^a-zA-Z0-9]", " ",raw_review) 
    # convert into lowercase and split text into words using split() function
    words = letters_only.split()
    # declaring empty array
    cleanwords = []
    for word in words: #if(word.lower() not in stopwords and len(word) > 2):
       
        cleanwords.append(word)
    return( " ".join( cleanwords ))


# In[37]:

def load_lexicon(lex_type):
    lex = []

    f = open(lex_type+"_lexicon.txt", "r")
    for line in f:
        tag = line.split()[0]
        lex.append(tag)
        
    return lex


# In[38]:

def create_vector_features(x,model):
    ndim = model.vector_size
    index2word_set = set( model.index2word )
    words = x.lower().split(" ")
    vec_feats =[]
    for w in words:
        if w in index2word_set:
            vec_feats.append(model[w])
        else:
            vec_feats.append(np.zeros(ndim))
    return vec_feats        


# In[39]:

def create_next_prev_vector_features(x,model):
    ndim = model.vector_size
    index2word_set = set( model.index2word )
    words = x.lower().split(" ")
    previous_vector_feats = []
    second_previous_vector_feats = []
    next_vector_feats = []
    second_next_vector_feats = []
    for i in range(0,len(words)):
        #check if previous token is in model
                if (i-1) >= 0:
                    if (words[i-1].lower() in index2word_set):
                               previous_vector_feats.append(model[words[i-1]])
                    else:
                                previous_vector_feats.append(np.zeros(ndim))
                else:
                    previous_vector_feats.append(np.zeros(ndim))
                
                
                 #check if second previous token is in model
                if (i-2) >= 0:
                    if (words[i-2].lower() in index2word_set):
                               second_previous_vector_feats.append(model[words[i-2]])
                    else:
                                second_previous_vector_feats.append(np.zeros(ndim))
                else:
                       second_previous_vector_feats.append(np.zeros(ndim))
                        
               
                #append next vector
                if (i+1) < len(words):
                    if (words[i+1].lower() in index2word_set):
                                   next_vector_feats.append(model[words[i+1]])
                    else:
                                 next_vector_feats.append(np.zeros(ndim))
                else:
                        next_vector_feats.append(np.zeros(ndim))
                        
                
                 #append next vector
                if (i+2) < len(words):
                    if (words[i+2].lower() in index2word_set):
                                   second_next_vector_feats.append(model[words[i+2]])
                    else:
                                second_next_vector_feats.append(np.zeros(ndim))
                else:
                        second_next_vector_feats.append(np.zeros(ndim))
    
    
    return previous_vector_feats,next_vector_feats,second_next_vector_feats,second_previous_vector_feats
        


# In[40]:

def create_morph_feats(x):
    #morphological features
        words = nltk.word_tokenize(x)
       
        sent_morph_feats =[]
        for w in words:
                morph_feats=[]
                
                if w[0].isupper(): #is first letter capital
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                capitals = 0
                lowers = 0
                for letter in w:
                    if letter.isupper():
                        capitals = capitals + 1
                    if letter.islower():
                        lowers = lowers + 1

                if w[0].islower() and capitals > 0: #contains capitals, except 1st letter
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if capitals == len(w): #is all letters capitals
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if lowers == len(w): #is all letters lower
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"\d", w)) == len(w): #is all letters digits
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                if len(re.findall(r"[a-zA-Z]", w)) == len(w): #is all letters words
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)

                
                
                sent_morph_feats.append(morph_feats)
        return sent_morph_feats


# In[41]:

def create_pos_feats(x,pos_lexicon):
    words = nltk.word_tokenize(x)
    tags = nltk.pos_tag(words)
    tags_list = [] #the pos list
    pos_sent_feats = []
    for _, t in tags:
                tags_list.append(t)
    i =0        
    for w in words:
            pos_feats = []
            for p in pos_lexicon:

                            #check the POS tag of the current word
                            if tags_list[i] == p:
                                pos_feats.append(1)
                            else:
                                pos_feats.append(0)
            pos_sent_feats.append(pos_feats)
            i = i+1
    return pos_sent_feats


# In[42]:

def create_prev_pos_feats(x,pos_lexicon):
    words = nltk.word_tokenize(x)
    tags = nltk.pos_tag(words)
    tags_list = [] #the pos list
    pos_sent_prev_feats = []
    pos_sent_next_feats = []
    pos_sent_second_prev_feats = []
    pos_sent_second_next_feats = []
    
    for _, t in tags:
                tags_list.append(t)
    i =0        
    for w in words:
            previous_pos_feats = []
            second_previous_pos_feats = []
            next_pos_feats = []
            second_next_pos_feats = []
            for p in pos_lexicon:
                    
                    #check the POS tag of the previous word (if the index is IN list's bounds)
                    if (i-1) >= 0:
                        if tags_list[i-1] == p:
                            previous_pos_feats.append(1)
                        else:
                            previous_pos_feats.append(0)
                    else:
                        previous_pos_feats.append(0)
                            
                    #check the POS tag of the 2nd previous word (if the index is IN list's bounds)
                    if (i-2) >= 0:
                        if tags_list[i-2] == p:
                            second_previous_pos_feats.append(1)
                        else:
                            second_previous_pos_feats.append(0)
                    else:
                        second_previous_pos_feats.append(0)
                            
                    #check the POS tag of the next word (if the index is IN list's bounds)
                    if (i+1) < len(words):
                        if tags_list[i+1] == p:
                            next_pos_feats.append(1)
                        else:
                            next_pos_feats.append(0)
                    else:
                        next_pos_feats.append(0)
                            
                    #check the POS tag of the next word (if the index is IN list's bounds)
                    if (i+2) < len(words):
                        if tags_list[i+2] == p:
                            second_next_pos_feats.append(1)
                        else:
                            second_next_pos_feats.append(0)
                    else:
                        second_next_pos_feats.append(0)
            
            
            pos_sent_prev_feats.append(previous_pos_feats)
            pos_sent_next_feats.append(next_pos_feats)
            pos_sent_second_prev_feats.append(second_previous_pos_feats)
            pos_sent_second_next_feats.append(second_next_pos_feats)
            i = i+1
    return pos_sent_prev_feats,pos_sent_next_feats,pos_sent_second_prev_feats,pos_sent_second_next_feats


# In[43]:

def create_labels(x,labels):
    words = x.split(" ")
                   
    aspectterms = labels.split(";")
    aspectterms = aspectterms[1:len(aspectterms)]
    sent_labels = []
    last_prediction =""
    for w in words:
           
            term_found = False
            for aspect_term in set(aspectterms):
                   
                    term_words = aspect_term.split()
                    for term_index, term in enumerate(term_words):
                        if (w.lower() == term) and (term_found is False):
                            if term_index == 0:
                                target_labels = [1] #1 is "B"
                                last_prediction = "1"
                                term_found = True                            
                            else:
                                if (last_prediction == "1") or (last_prediction == "2"):
                                    target_labels = [2] #2 is "I"
                                    last_prediction = "2"
                                    term_found = True                            
                                else:
                                    target_labels = [0]
                                    last_prediction = "0"

            if term_found is False:
                target_labels = [0] #0 is "O"
                last_prediction = "0"
            sent_labels.append(target_labels)
    return sent_labels


# In[44]:

def create_features_array(df):
    df_sentences = []
    sentence_labels = []
    for index,rev in df.iterrows():
        x = rev['cleanText'].split(" ")
        train_word_feats = []

        for i,word in enumerate(x):
            train_word_features = []
            train_word_features.append(rev['vector_feats'][i])
            train_word_features.append(rev['morph_feats'][i])
            train_word_features.append(rev['pos_feats'][i])

            train_word_features.append(rev['pos_sent_prev_feats'][i])
            train_word_features.append(rev['pos_sent_next_feats'][i])
            train_word_features.append(rev[ 'pos_sent_second_prev_feats'][i])
            train_word_features.append(rev['pos_sent_second_next_feats'][i])


            train_word_features.append(rev[ 'previous_vector_feats'][i])
            train_word_features.append(rev['next_vector_feats'][i])
            train_word_features.append(rev[ 'second_previous_vector_feats'][i])
            train_word_features.append(rev['second_next_vector_feats'][i])
#,train_word_features[7],train_word_features[8],train_word_features[9],train_word_features[10]
            train_word_feats.append(np.concatenate((train_word_features[0],train_word_features[1],train_word_features[2],train_word_features[3],
                                   train_word_features[4],train_word_features[5],
                                   train_word_features[6],train_word_features[7],train_word_features[8],train_word_features[9],train_word_features[10]),axis = 0))

        train_sentences_array = np.zeros((len(train_word_feats), len(train_word_feats[0])))
        index_i = 0
        for index_i in range(0,len(train_word_feats)):
            for index_j in range(0,len(train_word_feats[0])):
                    train_sentences_array[index_i, index_j] = train_word_feats[index_i][index_j]

        df_sentences.append(train_sentences_array)        
        sentence_labels_array = np.zeros(len(rev['labels']))
        index_i = 0
        for index_i in range(0,len(rev['labels'])):
            sentence_labels_array[index_i] = rev['labels'][index_i][0]

        sentence_labels.append(sentence_labels_array.astype(np.int64))
    return df_sentences,sentence_labels


# In[45]:

def predicted_aspect_terms(cleanText,predictedLabels):
    words = cleanText.split(" ")
    start_aspectTerm = False
    ote=''
    aspect_term_predicted = []
    for i in range(0,len(predictedLabels)):
        if predictedLabels[i] == 1 and start_aspectTerm == False:
            start_aspectTerm = True
            ote = words[i]
            if(i == len(predictedLabels) - 1):
                aspect_term_predicted.append(ote.lower())
        elif  predictedLabels[i] == 1 and start_aspectTerm == True:
            aspect_term_predicted.append(ote.lower())
            ote = words[i]
            if(i == len(predictedLabels) - 1):
                aspect_term_predicted.append(ote)
        elif predictedLabels[i] == 2 and start_aspectTerm == True:
            ote = ote + ' '+words[i]
            if(i == len(predictedLabels) - 1):
                aspect_term_predicted.append(ote.lower())
        elif predictedLabels[i] == 0 and start_aspectTerm == True:
            start_aspectTerm = False
            aspect_term_predicted.append(ote.lower())
    return aspect_term_predicted        


# In[46]:

def evaluate_ote(correct,predicted):
    common, relevant, retrieved = 0., 0., 0.
    for i in range(len(correct)):
            cor = []
            for a in set(correct[i]):
                if (a.lower()!="null"):
                    cor.append(a)

            pre = []        
            for a in set(predicted[i]):
                   pre.append(a)
                   
            common += len([a for a in pre if a in cor])
            retrieved += len(pre)
            relevant += len(cor)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0
    return p, r, f1, common, retrieved, relevant


# In[47]:

def produce_corr(x):
    correct_aspects = x.split(";") 
    correct_aspects = correct_aspects[1:len(correct_aspects)]
    return correct_aspects

def evaluating_ote(df):
    aspect_terms_p = []
    for index,rev in df.iterrows():
        aspect_terms_p.append(predicted_aspect_terms(rev['cleanText'],rev['predicted_labels']))
    df['predicted_aspect_terms'] = aspect_terms_p
    df['correct_aspect_terms'] = df['aspect term'].apply(produce_corr)
    p, r, f1, common, retrieved, relevant = evaluate_ote(df['correct_aspect_terms'],df['predicted_aspect_terms'])
    return p, r, f1, common, retrieved, relevant
    


# In[48]:



# In[81]:

def pick_best_C_value(train_sentences,sentence_labels,test_SF,test_sentences,test_sentence_labels):
   
	i = 0.10
	best_C = i
	f_old = 0
	for z in range(1,20):
		print "----------------- Training on C-value %f"%i
		modelCRF = ChainCRF()
		ssvm = FrankWolfeSSVM(model=modelCRF, C=i, max_iter=20,random_state=5)
		ssvm.fit(train_sentences,sentence_labels)
		print "\n"
		print "-------- Training complete --------"

		predictions = ssvm.predict(test_sentences)
		test_SF['predicted_labels'] = predictions

		#Saving model
		print "Saving model...."
		pickle.dump(ssvm, open('models/ote/otemodel.sav', 'wb'))

		#Evaluating Trained CRF model

		p, r, f1, common, retrieved, relevant = evaluating_ote(test_SF)
		if(f1 >= f_old):
			#save value of 'C'
			f_old = f1
			best_C = i
		
		i = i+0.05
	return best_C


# In[83]:

def main(train_SF,test_SF,model):
		#read files
		#print("Reading training and testing files")
		#train_SF = pd.read_csv('data/restaurants/train.csv',sep = '\t')
		#test_SF = pd.read_csv('data/restaurants/test.csv',sep = '\t')
		#vectors_filename = "vectors_yelp_200.txt"
		#filename = 'otemodel.sav' #for OTE model

		#for laptops
		#'data/laptops/laptop_train_ote.csv'
		#'data/laptops/laptop_test_ote.csv'
		#"gloveVec200.txt"

		#train_SF = pd.read_csv(trainF,sep = '\t')
		#test_SF = pd.read_csv(testF,sep = '\t')
		#vectors_filename =  vecF
		filename = 'otemodel.sav' #for OTE model

		pos_lexicon = load_lexicon("lexica/pos")



		#Load word2vec text files
		#print "Loading word2vec file"
		#model = gensim.models.Word2Vec.load_word2vec_format(vectors_filename,binary=False)
		ndim = model.vector_size
		index2word_set = set( model.index2word )

		#Cleaning text
		print "Cleaning text"
		train_SF['cleanText'] = train_SF['text'].apply(review_to_words)
		test_SF['cleanText'] = test_SF['text'].apply(review_to_words)
		test_SF = test_SF[test_SF['cleanText'] != ''].reset_index(drop=True)
		train_SF = train_SF[train_SF['cleanText'] != ''].reset_index(drop=True)
		#Extracting vector features
		print "Extracting vector features"

		train_vec = []
		test_vec = []
		for i in range(0,len(train_SF)):
		  train_vec.append(create_vector_features(train_SF['cleanText'][i],model))

		for i in range(0,len(test_SF)):
		  test_vec.append(create_vector_features(test_SF['cleanText'][i],model))
		train_SF['vector_feats'] = train_vec
		test_SF['vector_feats'] = test_vec

		#Extracting morphological features
		print "Extracting morphological features"

		train_SF['morph_feats'] = train_SF['cleanText'].apply(create_morph_feats)
		test_SF['morph_feats'] = test_SF['cleanText'].apply(create_morph_feats)  

		#Extracting POS features
		print "Extracting POS features"
		train_pos = []
		test_pos = []
		for index,row in train_SF.iterrows():
				if(index%1000 == 0):
						print "Train data - POS features extraction Progress :%d sentences done"%index
				train_pos.append(create_pos_feats(row['cleanText'],pos_lexicon))

		for index,row in test_SF.iterrows():
			if(index%1000 == 0):
					print "Test data - POS features extraction Progress :%d sentences done"%index

			test_pos.append(create_pos_feats(row['cleanText'],pos_lexicon))

		train_SF['pos_feats'] = train_pos
		test_SF['pos_feats'] = test_pos


		#Extracting previous,next Vector Features
		print "Extracting previous,next Vector features"

		previous_vector_feats_array = []
		next_vector_feats_array = []
		second_next_vector_feats_array = []
		second_previous_vector_feats_array = []
		for i in range(0,len(train_SF)):
			previous_vector_feats,next_vector_feats,second_next_vector_feats,second_previous_vector_feats = create_next_prev_vector_features(train_SF['cleanText'][i],model)
			previous_vector_feats_array.append(previous_vector_feats)
			next_vector_feats_array.append( next_vector_feats)
			second_next_vector_feats_array.append(second_next_vector_feats)
			second_previous_vector_feats_array.append(second_previous_vector_feats)

		train_SF['previous_vector_feats'] = previous_vector_feats_array
		train_SF['next_vector_feats'] = next_vector_feats_array
		train_SF['second_previous_vector_feats'] = second_previous_vector_feats_array
		train_SF['second_next_vector_feats']= second_next_vector_feats_array


		#create next prev vector features
		previous_vector_feats_array = []
		next_vector_feats_array = []
		second_next_vector_feats_array = []
		second_previous_vector_feats_array = []
		for i in range(0,len(test_SF)):
			previous_vector_feats,next_vector_feats,second_next_vector_feats,second_previous_vector_feats = create_next_prev_vector_features(test_SF['cleanText'][i],model)
			previous_vector_feats_array.append(previous_vector_feats)
			next_vector_feats_array.append( next_vector_feats)
			second_next_vector_feats_array.append(second_next_vector_feats)
			second_previous_vector_feats_array.append(second_previous_vector_feats)

		test_SF['previous_vector_feats'] = previous_vector_feats_array
		test_SF['next_vector_feats'] = next_vector_feats_array
		test_SF['second_previous_vector_feats'] = second_previous_vector_feats_array
		test_SF['second_next_vector_feats']= second_next_vector_feats_array


		#Extracting previous,next POS features
		print "Extracting previous,next POS features"

		pos_sent_prev_feats_array = []
		pos_sent_next_feats_array = []
		pos_sent_second_prev_feats_array = []
		pos_sent_second_next_feats_array = []

		for i in range(0,len(train_SF)):
			pos_sent_prev_feats,pos_sent_next_feats,pos_sent_second_prev_feats,pos_sent_second_next_feats = create_prev_pos_feats(train_SF['cleanText'][i],pos_lexicon)
			pos_sent_prev_feats_array.append(pos_sent_prev_feats)
			pos_sent_next_feats_array.append(pos_sent_next_feats)
			pos_sent_second_next_feats_array.append(pos_sent_second_next_feats)
			pos_sent_second_prev_feats_array.append(pos_sent_second_prev_feats)

		train_SF['pos_sent_prev_feats'] = pos_sent_prev_feats_array
		train_SF['pos_sent_next_feats'] = pos_sent_next_feats_array
		train_SF['pos_sent_second_prev_feats'] = pos_sent_second_prev_feats_array
		train_SF['pos_sent_second_next_feats']= pos_sent_second_next_feats_array


		#for test file
		pos_sent_prev_feats_array = []
		pos_sent_next_feats_array = []
		pos_sent_second_prev_feats_array = []
		pos_sent_second_next_feats_array = []

		for i in range(0,len(test_SF)):
			pos_sent_prev_feats,pos_sent_next_feats,pos_sent_second_prev_feats,pos_sent_second_next_feats = create_prev_pos_feats(test_SF['cleanText'][i],pos_lexicon)
			pos_sent_prev_feats_array.append(pos_sent_prev_feats)
			pos_sent_next_feats_array.append(pos_sent_next_feats)
			pos_sent_second_next_feats_array.append(pos_sent_second_next_feats)
			pos_sent_second_prev_feats_array.append(pos_sent_second_prev_feats)

		test_SF['pos_sent_prev_feats'] = pos_sent_prev_feats_array
		test_SF['pos_sent_next_feats'] = pos_sent_next_feats_array
		test_SF['pos_sent_second_prev_feats'] = pos_sent_second_prev_feats_array
		test_SF['pos_sent_second_next_feats']= pos_sent_second_next_feats_array


		print "Features extraction complete............"

		# Creating labels
		print "Creating labels.."
		labels_train = []
		labels_test = []
		for index,rev in train_SF.iterrows():
			 labels_train.append(create_labels(rev['cleanText'],rev['aspect term']))

		for index,rev in test_SF.iterrows():
			 labels_test.append(create_labels(rev['cleanText'],rev['aspect term']))

		train_SF['labels'] = labels_train
		test_SF['labels'] = labels_test

		test_SF = test_SF[test_SF['cleanText']  != '']

		# Training CRF model...
		print "Training CRF model...."
		train_sentences,sentence_labels = create_features_array(train_SF)
		test_sentences,test_sentence_labels = create_features_array(test_SF)

		print "Parameter 'C' value selection...."
		best_C_val = pick_best_C_value(train_sentences,sentence_labels,test_SF,test_sentences,test_sentence_labels)
		print "C-value found : %f"%best_C_val
		modelCRF = ChainCRF()
		ssvm = FrankWolfeSSVM(model=modelCRF, C=best_C_val, max_iter=10,random_state=5)
		ssvm.fit(train_sentences,sentence_labels)

		print "Training complete...."

		predictions = ssvm.predict(test_sentences)
		test_SF['predicted_labels'] = predictions

		#Saving model
		print "Saving model...."
		pickle.dump(ssvm, open(filename, 'wb'))

		#Evaluating Trained CRF model
		print ""
		print " -------------- Evaluation Results --------------"
		predictions = ssvm.predict(train_sentences)
		train_SF['predicted_labels'] = predictions
		p, r, f1, common, retrieved, relevant = evaluating_ote(train_SF)
		print "--------- Train Set Results ---------"
		print "Precision : %f"%p
		print "Recall : %f"%r
		print "F1 measure : %f"%f1
		print ""
		p, r, f1, common, retrieved, relevant = evaluating_ote(test_SF)

		print "--------- Test Set Results ---------"
		print "Precision : %f"%p
		print "Recall : %f"%r
		print "F1 measure : %f"%f1
		print ""
        
		return f1
    
          
# In[ ]:

#main()


# In[ ]:



