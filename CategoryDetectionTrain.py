
# coding: utf-8

# In[1]:

#import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
import numpy as np
#from collections import Counter
#import operator
import sklearn
import nltk
import pandas as pd
import xgboost
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import gensim
from gensim.models import Word2Vec
import numpy as np
import re
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import pickle
# In[6]:

stopwords =  nltk.corpus.stopwords.words('english')
not_remove = ['not','very','few','more','only','nor','but','don','t','and','with','no']
for word in not_remove:
    stopwords.remove(word)
def review_to_wordlist( review, remove_stopwords=True ):
    # remove HTML
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    if remove_stopwords:
         words = [w for w in words if not w in stopwords]
            
    return ( words )

def review_to_words( raw_review ):
    #remove non alphanumeric characters
    letters_only = re.sub("\'", "",raw_review) 
    letters_only = re.sub("[^a-zA-Z]", " ",letters_only) 
    # convert into lowercase and split text into words using split() function
    words = letters_only.lower().split()
    # declaring empty array
    cleanwords = []
    for word in words:
        if(word not in stopwords):
            cleanwords.append(word)
    return( " ".join( cleanwords ))


# In[7]:

nan_words = {}

def makeFeatureVec( words, model, num_features, index2word_set ):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            if np.isnan( model[ word ] ).any():
                if word in nan_words:
                    nan_words[ word ] += 1
                else:
                    nan_words[ word ] = 1
    
            featureVec = np.add(featureVec,model[word])
    if nwords != 0:
        featureVec = np.divide(featureVec,nwords)

    return featureVec

def getAvgFeatureVecs(reviews, model, num_features, index2word_set ):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    for review in reviews:
       if counter % 1000 == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features, index2word_set )
       counter = counter + 1.

    return reviewFeatureVecs



# In[14]:



def computeCategorySimilarity(x,cat_words_vectors,model,index2word_set):
    
    words = x.split(" ")
    words_vectors = []
    similarity_from_category = np.zeros((len(cat_words_vectors),),dtype="float32")
    if(x != ''):
        for w in words:
            if w in index2word_set:
                words_vectors.append(model[w])
                
        #computing word vector similarity from each category 
        if(len(words_vectors) > 0):
               for cat in cat_words_vectors:
                    np.append(similarity_from_category,cosine_similarity(words_vectors,cat).max())
        else:
            similarity_from_category =np.zeros(len(cat_words_vectors))
    else:
        similarity_from_category= np.zeros(len(cat_words_vectors))
    return similarity_from_category
    

    
# # Training 'n' binary xgboost classifiers

def save_model(model,directory,model_name):
			print 'Saving model...'
			joblib.dump(model, directory+'/'+model_name+'.pkl')
			return True

f1_scorer = make_scorer(f1_score, pos_label=1)

def return_prob_positive_class(ans):
    result = []
    for i in range(0,len(ans)):
        result.append(ans[i][1])
    return result

def train_xgboost_model_foreachCategory(train_feats,test_feats,test_SF,categories,train_SF):
    x_params = { 'max_depth':range(3,10,2),
             'n_estimators': [80,100],
            'subsample':[i/100.0 for i in range(60,90,5)],
            'min_child_weight' :range(1,6,2),
            'colsample_bytree' :[i/100.0 for i in range(60,90,5)],
            'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
            'learning_rate':[0.1],
            'objective':['binary:logistic'],
            'scale_pos_weight' : [1,5,10]
           }
    xgb_model = xgboost.XGBClassifier()
    for cat in categories:
				print ""
				print "----- Training Classifier for %s Class -------- " %cat
				clf = RandomizedSearchCV(xgb_model, x_params,cv = 3,scoring=f1_scorer)
				clf.fit(train_feats,train_SF[cat])
				#saved = save_model(clf,'models/acd','model_'+cat)
				#print(clf.best_score_)
				#print(clf.best_params_)
				
				print "Saving model ... "
				filename = 'models/acd/model_'+ str(cat)+'.sav'
				pickle.dump(clf.best_estimator_, open(filename, 'wb'))
				predict_col = 'predicted_prob_'+cat
				test_SF[predict_col] = return_prob_positive_class(clf.predict_proba(test_feats))
				print '-------------------------------------'
				
    return test_SF


# Aspect Category Detection
def category_detection(correct,predicted):
    common, relevant, retrieved = 0., 0., 0.
    for i in range(len(correct)):
        cor = set(correct[i])
        # Use set to avoid duplicates (i.e., two times the same category)
        pre = predicted[i]
        common += len([c for c in pre if c in cor])
        retrieved += len(pre)
        relevant += len(cor)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return p, r, f1, common, retrieved, relevant


# In[50]:

def evaluate(test,categories):
    predictedLabels = []
    correctLabels = test['mod_category']
    colnames = []
    for cat in categories:
        colnames.append('predicted_prob_'+cat)
   
    for index,row in test.iterrows():
        r_labels =[]
        for c in colnames:
               if(row[c] >= 0.45):
                    sd = re.sub('predicted_prob_', "",c)
                    r_labels.append(sd)
        
        predictedLabels.append(r_labels)
    p, r, f1, common, retrieved, relevant = category_detection(correctLabels,predictedLabels)
    return p, r, f1, common, retrieved, relevant,predictedLabels,correctLabels


# In[ ]:
def main(train_SF,test_SF,model):
	 # Read train and Test CSV Files
	#print "Reading train and test files"    
	#train_SF = pd.read_csv(trainF,sep = '\t')
	#test_SF = pd.read_csv(testF,sep = '\t')
	#vectors_filename = vecF
	#model = gensim.models.Word2Vec.load_word2vec_format(vectors_filename,binary=False)
	ndim = model.vector_size

	print "Cleaning text..."
	train_SF['cleanText'] = train_SF['text'].apply(review_to_words)
	test_SF['cleanText'] = test_SF['text'].apply(review_to_words)
	test_SF = test_SF[test_SF['cleanText'] != ''].reset_index(drop=True)
	train_SF = train_SF[train_SF['cleanText'] != ''].reset_index(drop=True)


	print "Extracting vector features..."
	index2word_set = set( model.index2word )
	clean_train_reviews = []
	for review in train_SF['cleanText']:
		clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

	trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model,ndim, index2word_set )

	clean_test_reviews = []
	for review in test_SF['cleanText']:
		clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

	testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, ndim, index2word_set )

	 # ## Extract Distinct category labels 

	print "Extracting category label..."

	categories = set()
	train_SF['category'].str.lower().str.split().apply(categories.update)

	#add binary columns
	train_SF['mod_category'] = train_SF['category'].apply(lambda x : x.strip().split(' '))
	test_SF['mod_category'] = test_SF['category'].apply(lambda x : x.strip().split(' '))

	for cat in categories:
		train_SF[cat] = train_SF.category.apply(lambda x: 1 if cat in x else 0)
		test_SF[cat] = test_SF.category.apply(lambda x: 1 if cat in x else 0)

	## Feature Extraction

	# In[12]:

	cat_words_vectors = []
	for cat in categories:
				similar_words_vec = []
				similar_words = model.most_similar(cat)
				for word in similar_words:
					similar_words_vec.append(model[word[0]])
				cat_words_vectors.append(similar_words_vec)

	print "Extracting Category Similarity Feature"

	cs_train = np.zeros((len(train_SF),len(categories)),dtype="float32")
	i = 0
	for rev in train_SF['cleanText']:
		cs_train[i] = computeCategorySimilarity(rev,cat_words_vectors,model,index2word_set)
		i = i+1

	cs_test = np.zeros((len(test_SF),len(categories)),dtype="float32")
	i = 0
	for rev in test_SF['cleanText']:
		cs_test[i] = computeCategorySimilarity(rev,cat_words_vectors,model,index2word_set)
		i = i+1

	print ""
	print "----------------------------------------------------------------"
	print "Training %d binary classifiers using XGBOOST"%len(categories)
	print "----------------------------------------------------------------"

	train_feats =np.column_stack((trainDataVecs,cs_train))
	test_feats = np.column_stack((testDataVecs,cs_test))

	t = train_xgboost_model_foreachCategory(train_feats,test_feats,test_SF,categories,train_SF)
	print "----------------------------------------------------------------"
	print "\t Evaluation on Test Data \t\t"
	print "----------------------------------------------------------------"

	p, r, f1, common, retrieved, relevant,predictedLabels,correctLabels = evaluate(t,categories)

	print "Precision : %f "%p
	print "Recall    : %f "%r
	print "F-Measure : %f "%f1
	print ""
	return f1
#main()

    
    




