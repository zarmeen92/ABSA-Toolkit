
# coding: utf-8

# In[1]:
from __future__ import division
#import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
import numpy as np
#from collections import Counter
#import operator
import sklearn
import nltk
import pandas as pd
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import gensim
from gensim.models import Word2Vec
import numpy as np
import re
import os
from sklearn.externals import joblib
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree, ParentedTree
from sklearn.model_selection import ShuffleSplit
import os
import itertools
#from nltk.internals import find_jars_within_path
#os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk1.8.0_131'
#os.environ['STANFORD_PARSER'] = 'stanford-parser-full-2016-10-31/stanford-parser-full-2016-10-31/stanford-parser.jar'
#os.environ['STANFORD_MODELS'] = 'stanford-parser-full-2016-10-31/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar'
#parser=StanfordParser(model_path="stanford-parser-full-2016-10-31/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#parser._classpath = tuple(find_jars_within_path('stanford-parser-full-2016-10-31/'))
# In[29]:
taglist = ['JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN']

def recreate_df(train_pd):
    text = []
    ate = []
    cat = []
    pol = []
    
    for i,row in train_pd.iterrows():
        aspect_terms = row['aspect term'].split(';')
        aspect_terms = filter(None,aspect_terms)
        categories = row['category'].split(' ')
        categories = filter(None,categories)
        polarity = row['polarity'].split(' ')
        polarity = filter(None,polarity)
        if(len(polarity) == len(categories) and len(categories) == len(aspect_terms) ):
            for j in range(0,len(aspect_terms)):
                text.append(row['text'])
                ate.append(aspect_terms[j])
                cat.append(categories[j])
                pol.append(polarity[j])
    
    new_train = pd.DataFrame({'text' : text,'aspect term' : ate,'category' : cat,'polarity' :pol })
    return new_train            
    


# In[32]:


def load_polarity_lexicon(filename):
    lex = []
    scores=[]
    f = open(filename, "r")#'saif_lex/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt'
    for line in f:
        l = line.split()
        tag = l[0]
        score = l[1]
        if(len(re.findall('[_NEG]',tag)) == 0):
            lex.append(tag)
            scores.append(score)
        
    return lex,scores




# In[37]:

stopwords =  nltk.corpus.stopwords.words('english')
not_remove = ['not','very','few','more','only','nor','but','don','t','and','with','no']
for word in not_remove:
    stopwords.remove(word)


# In[38]:

# reading stop word list from nltk
import nltk

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



neg_words = ['not','dont','didnt','wont','havent','don','nor','never','no','cannot','nothing','non','doesnot','didnot','didn']
def create_lexicon_feats(x,yelp_lex,yelp_scores):
	x = x.lower()
	x = re.sub('\.',' ',x).strip()
	count = 0
	if x.strip() != '':
		words = nltk.word_tokenize(x)
		tags = nltk.pos_tag(words)
		#print tags
		
		for j in range(0,len(tags)):
			if (tags[j][0] not in neg_words and tags[j][1] in taglist):
				for i in range(0,len(yelp_lex)):
					
					 if tags[j][0] == yelp_lex[i]:
							if (j > 0 and tags[j-1][0] in neg_words) or (j>1 and tags[j-2][0] in neg_words) or (j>2 and tags[j-3][0] in neg_words):
								count = count + (-1)*float(yelp_scores[i])
							else:
								count = count + float(yelp_scores[i])
					   
							break
	   
	return count
    


# In[42]:

def find_near_words(aspect_term,text,window):
    if(aspect_term != 'null'):
        aspect_term = re.sub('[\',()]','',aspect_term)
        aspect_term = re.sub("[^a-zA-Z]", " ",aspect_term).lower()
        
        multi_asp = aspect_term.split(' ')
        aspect_term = ''
        for w in multi_asp:
            if(len(w) != 0):
                if w not in stopwords:
                    aspect_term = aspect_term+' '+w
        #text = re.sub("[^a-zA-Z]", " ",text).lower()
        aspect_term = aspect_term.strip()
        new_text = text.replace(aspect_term,aspect_term.replace(' ',''))
        new_aspect_term = aspect_term.replace(' ','')
        left_words = ''
        right_words =''
        sent = new_text.split(' ')
        index_at=-1
        for i in range(0,len(sent)):
            if sent[i].find(new_aspect_term) != -1:
                index_at = i
                break
        #index_at = sent.index(new_aspect_term)
        if(index_at >= 0):
            j = index_at - window
            for count in range(0,window):
                if(j >= 0):
                    left_words = left_words + ' ' + sent[j]
                j = j + 1

            j = index_at + 1
            for count in range(0,window):
                if(j < len(sent)):
                    right_words = right_words + ' ' + sent[j]
                j = j + 1

            phrase = left_words + ' ' +aspect_term + right_words
        else:
            phrase = text
    else:
        phrase = text
        
    return phrase
    

nan_words = {}


# In[57]:

def return_category_vec(x,model):
    return model[x]


nan_words = {}

def review_to_wordlist( review, remove_stopwords=True ):
    # remove HTML
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    if remove_stopwords:
         words = [w for w in words if not w in stopwords]
            
    return ( words )
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

def create_lexicon_features_df(df,window,yelp_lex,yelp_scores):
    pos_feats = []
    for ind,rev in df.iterrows():
        pos_feats.append(create_lexicon_feats(rev['nearwords_window'+window],yelp_lex,yelp_scores))

    return pos_feats


def create_binary_columns(df,polarities):
    for polarity in polarities:
        df['is_'+polarity] = df['polarity'] == polarity
        df['is_'+polarity] =  df['is_'+polarity].astype(int)
    return df

# # Training

# In[94]:

f1_scorer = make_scorer(f1_score, pos_label=1)

def save_model(model,directory,model_name):
    print 'Saving model...'
    joblib.dump(model, directory+'/'+model_name+'.pkl')

def return_prob_positive_class(ans):
    result = []
    for i in range(0,len(ans)):
        result.append(ans[i][1])
    return result


# In[108]:

def train_xgboost_model_foreachPolarity(train_feats,test_feats,test_SF,polarities,train_SF):
    x_params = { 
            'max_depth':range(3,10,2),
            'n_estimators': [80,100],
            'objective':['binary:logistic'],
            'seed':[5]
           }
    xgb_model = xgboost.XGBClassifier()
    for pol in polarities:
				print ""
				print "----------------------------------------"
				print "Training Classifier for %s Class " %pol
				print "----------------------------------------"
				print ""
				cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=5)
				clf = GridSearchCV(xgb_model, x_params,cv = cv)
				clf.fit(train_feats,train_SF['is_'+pol])
				#save_model(clf.best_estimator_,'models/pd','model_'+pol)
				#print(clf.best_score_)
				#print(clf.best_params_)
				print "Saving model ..."
				filename = 'models/pd/model_'+ str(pol)+'.sav'
				pickle.dump(clf.best_estimator_, open(filename, 'wb'))
				predict_col = 'predicted_prob_'+pol
				test_SF[predict_col] = return_prob_positive_class(clf.predict_proba(test_feats))

				print '-------------------------------------'
				print " F- Score : %f"%clf.score(test_feats,test_SF['is_'+pol])
				print " Accuracy : %f"%accuracy_score(test_SF['is_'+pol],clf.predict(test_feats))
    return test_SF

	
def train_randomforest_model_foreachPolarity(train_feats,test_feats,test_SF,polarities,train_SF):
	x_params = { 'max_depth':range(3,10,2),
			 'n_estimators': [60,80,100],
			 'class_weight' : ['balanced'],
			 'random_state':[5]
		   }
	rf_model =  RandomForestClassifier()
	for pol in polarities:
				print ""
				print "----------------------------------------"
				print "Training Random Forest Classifier for %s Class " %pol
				print "----------------------------------------"
				print ""
				cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=5)
				clf = GridSearchCV(rf_model, x_params,cv = cv)
				clf.fit(train_feats,train_SF['is_'+pol])
				#save_model(clf.best_estimator_,'models/pd','model_'+pol)
				#print(clf.best_score_)
				#print(clf.best_params_)
				print "Saving model ..."
				filename = 'models/pd/model_'+ str(pol)+'.sav'
				pickle.dump(clf.best_estimator_, open(filename, 'wb'))
				predict_col = 'predicted_prob_'+pol
				test_SF[predict_col] = return_prob_positive_class(clf.predict_proba(test_feats))

				print '-------------------------------------'
				print " F- Score : %f"%clf.score(test_feats,test_SF['is_'+pol])
				print " Accuracy : %f"%accuracy_score(test_SF['is_'+pol],clf.predict(test_feats))
	return test_SF
def train_adaboost_model_foreachPolarity(train_feats,test_feats,test_SF,polarities,train_SF):
	x_params = { 
			 'n_estimators': [60,80,100],
		   
		   }
	ada_model =   AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))

	for pol in polarities:
				print ""
				print "----------------------------------------"
				print "Training ADA Boost Classifier for %s Class " %pol
				print "----------------------------------------"
				print ""
				clf = GridSearchCV(ada_model, x_params,cv = 3)
				clf.fit(train_feats,train_SF['is_'+pol])
				#save_model(clf.best_estimator_,'models/pd','model_'+pol)
				#print(clf.best_score_)
				#print(clf.best_params_)
				print "Saving model ..."
				filename = 'models/pd/model_'+ str(pol)+'.sav'
				pickle.dump(clf.best_estimator_, open(filename, 'wb'))
				predict_col = 'predicted_prob_'+pol
				test_SF[predict_col] = return_prob_positive_class(clf.predict_proba(test_feats))

				print '-------------------------------------'
				print " F- Score : %f"%clf.score(test_feats,test_SF['is_'+pol])
				print " Accuracy : %f"%accuracy_score(test_SF['is_'+pol],clf.predict(test_feats))
	return test_SF

def train_naiveBayes_model_foreachPolarity(train_feats,test_feats,test_SF,polarities,train_SF):
	for pol in polarities:
				print ""
				print "----------------------------------------"
				print "Training Naive Bayes Classifier for %s Class " %pol
				print "----------------------------------------"
				print ""
				clf = GaussianNB()
				clf.fit(train_feats,train_SF['is_'+pol])
				#save_model(clf.best_estimator_,'models/pd','model_'+pol)
				#print(clf.best_score_)
				#print(clf.best_params_)
				print "Saving model ..."
				filename = 'models/pd/model_'+ str(pol)+'.sav'
				pickle.dump(clf, open(filename, 'wb'))
				predict_col = 'predicted_prob_'+pol
				test_SF[predict_col] = return_prob_positive_class(clf.predict_proba(test_feats))

				print '-------------------------------------'
				print " F- Score : %f"%clf.score(test_feats,test_SF['is_'+pol])
				print " Accuracy : %f"%accuracy_score(test_SF['is_'+pol],clf.predict(test_feats))
	return test_SF
	
def train_svc_model_foreachPolarity(train_feats,test_feats,test_SF,polarities,train_SF):
	x_params = { 'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
			  'C':[i/100.0 for i in range(10,100,5)],
			  'probability':[True],
			 'class_weight' : ['balanced'],
			 'random_state':[5],
		   }
	svm_model =  SVC()
	for pol in polarities:
				print ""
				print "----------------------------------------"
				print "Training SVM Classifier for %s Class Using RandomizedSearchCV " %pol
				print "----------------------------------------"
				print ""
				cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=5)
				clf = GridSearchCV(svm_model, x_params,cv = cv)
				clf.fit(train_feats,train_SF['is_'+pol])
				#save_model(clf.best_estimator_,'models/pd','model_'+pol)
				#print(clf.best_score_)
				#print(clf.best_params_)
				print "Saving model ..."
				filename = 'models/pd/model_'+ str(pol)+'.sav'
				pickle.dump(clf.best_estimator_, open(filename, 'wb'))
				predict_col = 'predicted_prob_'+pol
				test_SF[predict_col] = return_prob_positive_class(clf.predict_proba(test_feats))

				print '-------------------------------------'
				print " F- Score : %f"%clf.score(test_feats,test_SF['is_'+pol])
				print " Accuracy : %f"%accuracy_score(test_SF['is_'+pol],clf.predict(test_feats))
	return test_SF

def compute_label(pos_prob,neg_prob,neu_prob=0):
	if(pos_prob >= neg_prob and pos_prob >= neu_prob):
		return "positive"
	elif(neg_prob > pos_prob and neg_prob >= neu_prob):
		return "negative"
	else:
		return "neutral"
	
    #if(pos_prob < 0.2 and neu_prob < 0.2 and neg_prob < 0.2):
    #    return "positive"
    
    #elif(abs(neu_prob - neg_prob) < 0.2 and pos_prob < 0.2):
    #    return "neutral"
    
    #elif(abs(neu_prob - pos_prob) < 0.2 and neg_prob < 0.2):
    #    return "neutral"
    
    #elif(neu_prob > pos_prob and neu_prob > neg_prob):
    #    return "neutral"
    
    #elif( pos_prob > neu_prob  and abs(neg_prob - pos_prob) > 0.2 and pos_prob > neg_prob):
    #    return "positive"
    #elif(neg_prob > pos_prob  and abs(neg_prob - pos_prob) > 0.2 and neg_prob > neu_prob):
    #    return "negative"
    #else:
    #    return "positive"
def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def get_labels_from_lca(ptree, lca_len, location):
    labels = []
    for i in range(lca_len, len(location)):
        labels.append(ptree[location[:i]].label())
    return labels

def findPath(ptree, text1, text2):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text1)
    leaf_index2 = leaf_values.index(text2)

    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)

    #find length of least common ancestor (lca)
    lca_len = get_lca_length(location1, location2)

    #find path from the node1 to lca

    labels1 = get_labels_from_lca(ptree, lca_len, location1)
    #ignore the first element, because it will be counted in the second part of the path
    result = labels1[1:]
    #inverse, because we want to go from the node to least common ancestor
    result = result[::-1]

    #add path from lca to node2
    result = result + get_labels_from_lca(ptree, lca_len, location2)
    return result

def compute_lexicon_score_constituency_parsing(text,aspectterm,yelp_lex,yelp_scores):
    x = text.lower()
    x= re.sub("\'", " ",x) 
    x = re.sub("[^a-zA-Z]", " ",x) 
    aspectterm = re.sub("\'", " ",aspectterm) 
    aspectterm = re.sub("[^a-zA-Z]", " ",aspectterm).strip() 
    print aspectterm
    print x
    if x.strip() != '':
        s = parser.raw_parse(x)
        tree = list(s)[0]
       

        words = nltk.word_tokenize(x)
        aspect = nltk.word_tokenize(aspectterm)[0]
        tags = nltk.pos_tag(words)
        #print tags
        count = 0
        for j in range(0,len(tags)):
            if (tags[j][0] not in neg_words and tags[j][1] in taglist):
                for i in range(0,len(yelp_lex)):

                     if tags[j][0] == yelp_lex[i]:
                        if aspect != 'null':  
                            distance_from_aspect =  len(findPath(tree,aspect,tags[j][0]))
                        else:
                            distance_from_aspect = 1
                        if distance_from_aspect != 0:
                            if(j >0):
                                if (j > 0 and tags[j-1][0] in neg_words) or (j>1 and tags[j-2][0] in neg_words) or (j>2 and tags[j-3][0] in neg_words):
                                    neg_handle = (-1)*float(yelp_scores[i])
                                    #if(neg_handle <0):
                                        #weighted_score = neg_handle*distance_from_aspect
                                    #else:
                                    weighted_score = neg_handle/distance_from_aspect
                                else:
                                    #if yelp_scores[i]<=0:
                                     #   weighted_score = float(yelp_scores[i])*distance_from_aspect
                                    #else:
                                    weighted_score = float(yelp_scores[i])/distance_from_aspect
                               
                            else:        
                                weighted_score = float(yelp_scores[i])/distance_from_aspect
                            count = count+weighted_score
                            #print 'Count %f'%count
                            #print 'Word %s'%tags[j][0]
                            #print 'Dict score %f'%float(yelp_scores[i])
                            #print 'Weighted by dist %f'%weighted_score
                            #print 'Disctance %d'%distance_from_aspect
                            #print '----'
                            break


        return count
    else:
        return 0
    
def get_distance(w1, w2,words):
    if w1 in words and w2 in words:
        w1_indexes = [index for index, value in enumerate(words) if value == w1]    
        w2_indexes = [index for index, value in enumerate(words) if value == w2]    
        distances = [abs(item[0] - item[1]) for item in itertools.product(w1_indexes, w2_indexes)]
        return {'min': min(distances), 'avg': sum(distances)/float(len(distances))}
def compute_lexicon_score_token_based(text,aspectterm,yelp_lex,yelp_scores):
	#print text
	#print aspectterm
	x = text.lower()
	x= re.sub("\'", " ",x) 
	x = re.sub("[^a-zA-Z]", " ",x) 
	aspectterm = re.sub("\'", " ",aspectterm) 
	aspectterm = re.sub("[^a-zA-Z]", " ",aspectterm).strip()
	make_aspectterm_one = aspectterm.split(" ")[0]
	#x = re.sub(aspectterm,make_aspectterm_one,x)
	#print make_aspectterm_one
	#print x
	if x.strip() != '':
		words = nltk.word_tokenize(x)
		tags = nltk.pos_tag(words)
		#print tags
		count = 0
		for j in range(0,len(tags)):
			if (tags[j][0] not in neg_words and tags[j][1] in taglist):
				for i in range(0,len(yelp_lex)):

					 if tags[j][0] == yelp_lex[i]:
						if make_aspectterm_one != 'null':  
							distance_from_aspect =  get_distance(make_aspectterm_one,tags[j][0],words)['min']
						else:
							distance_from_aspect = 1
						if distance_from_aspect != 0:
							if(j >0):
								if (j > 0 and tags[j-1][0] in neg_words) or (j>1 and tags[j-2][0] in neg_words) or (j>2 and tags[j-3][0] in neg_words):
									neg_handle = (-1)*float(yelp_scores[i])
									#if(neg_handle <0):
										#weighted_score = neg_handle*distance_from_aspect
									#else:
									weighted_score = neg_handle/distance_from_aspect
								else:
									#if yelp_scores[i]<=0:
									 #   weighted_score = float(yelp_scores[i])*distance_from_aspect
									#else:
									weighted_score = float(yelp_scores[i])/distance_from_aspect
							   
							else:        
								weighted_score = float(yelp_scores[i])/distance_from_aspect
							count = count+weighted_score
							#print 'Count %f'%count
							#print 'Word %s'%tags[j][0]
							#print 'Dict score %f'%float(yelp_scores[i])
							#print 'Weighted by dist %f'%weighted_score
							#print 'Disctance %d'%distance_from_aspect
							#print '----'
							break


		return count
	else:
		return 0
def training_clf(train_feats,test_feats,test_pd,polarities,train_pd,algCD):
	print "------- Training Sentiment Polarity Detection Classifiers --------"

	if algCD == 'Random Forest':
		test_pd = train_randomforest_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	elif algCD == 'Naive Bayes':
		test_pd = train_naiveBayes_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	elif algCD == 'SVM':
		test_pd = train_svc_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	elif algCD == 'ADA Boost':
		test_pd = train_adaboost_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	else:
		test_pd = train_xgboost_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)


	labels = []
	if 'is_neutral' in test_pd.columns:
		for ind,rev in test_pd.iterrows():
			labels.append(compute_label(rev['predicted_prob_positive'],rev['predicted_prob_negative'],rev['predicted_prob_neutral']))

	else:
		for ind,rev in test_pd.iterrows():
			labels.append(compute_label(rev['predicted_prob_positive'],rev['predicted_prob_negative']))


	test_pd['predictedLabels_multi'] = labels   

	acc = accuracy_score(test_pd['polarity'],test_pd['predictedLabels_multi'])
	fsco = f1_score(test_pd['polarity'],test_pd['predictedLabels_multi'],average='weighted')
	prec = precision_score(test_pd['polarity'],test_pd['predictedLabels_multi'],average='weighted')
	rec = recall_score(test_pd['polarity'],test_pd['predictedLabels_multi'],average='weighted')
	print(classification_report(test_pd['polarity'],test_pd['predictedLabels_multi'], target_names=test_pd['polarity'].unique()))

	print "--------------------------------------"
	print "\t Evaluation Results \t"
	print "--------------------------------------"

	print "Accuracy %f :"%acc
	print "F Score %f :"%fsco
	print "Precision Score %f :"%prec
	print "Recall Score %f :"%rec
	
	print ""
	return acc,fsco,prec,rec    
def compute_near_words(df,window):
		near_words = []
		for ind,rev in df.iterrows():
			near_words.append(find_near_words(rev['aspect term'],rev['cleanText'],window))
		return near_words

def main(train_pd,test_pd,model,lex_file,algCD):
	# Read train and Test CSV Files
	#print "Reading train and test files"    
	#train_pd = pd.read_csv('data/restaurants/train.csv',sep = '\t')
	#test_pd = pd.read_csv('data/restaurants/test.csv',sep = '\t')
	#vectors_filename = "vectors_yelp_200.txt"
	#lex_file = 'saif_lex/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt'
	#lex_file = 'saif_lex/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt'
	#lex_file = 'saif_lex/wnscores_inquirer.txt'
	#print "Loading Word2Vec Model..."
	#model = gensim.models.Word2Vec.load_word2vec_format(vectors_filename,binary=False)
	ndim = model.vector_size
	index2word_set = set(model.index2word)

	train_pd = recreate_df(train_pd)
	test_pd = recreate_df(test_pd)
	yelp_lex,yelp_scores = load_polarity_lexicon(lex_file)

	print "Cleaning text..."
	train_pd['cleanText'] = train_pd['text'].apply(review_to_words)
	test_pd['cleanText'] = test_pd['text'].apply(review_to_words)

	
	
	# Extracting category vector
	cat_vec = []
	for i,rev in train_pd.iterrows():
		cat_vec.append(return_category_vec(rev['category'],model))

	train_pd['category_vec'] = cat_vec
	cat_vec_test = []
	for i,rev in test_pd.iterrows():
		cat_vec_test.append(return_category_vec(rev['category'],model))

	test_pd['category_vec'] = cat_vec_test


	# creating vectors
	clean_train_reviews = []
	for review in train_pd['text']:
		clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

	trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model,ndim, index2word_set )

	clean_test_reviews = []
	for review in test_pd['text']:
		clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

	testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, ndim, index2word_set )

	print "Extracting token based distance score (Train set)..."
	token_distance_score = []
	for ind,rev in train_pd.iterrows():
		if ind % 100 == 0.:
		   print "Review %d of %d" % (ind, len(train_pd))
		token_distance_score.append(compute_lexicon_score_token_based(rev['text'],rev['aspect term'],yelp_lex,yelp_scores))
	train_pd['token_distance_score'] = token_distance_score

	print "Extracting token based distance score (Test set)"
	token_distance_score = []
	for ind,rev in test_pd.iterrows():
		if ind % 100 == 0.:
			print "Review %d of %d" % (ind, len(test_pd))
		token_distance_score.append(compute_lexicon_score_token_based(rev['text'],rev['aspect term'],yelp_lex,yelp_scores))
	test_pd['token_distance_score'] = token_distance_score


	# ## Create Binary Columns for Polarity Classes

	# In[75]:

	polarities = set(train_pd['polarity'])
	print "Creating binary Columns for each polarity...."
	train_pd = create_binary_columns(train_pd,polarities)
	test_pd = create_binary_columns(test_pd,polarities)

	#train_feats =np.column_stack((trainDataVecs,train_pd.yelp_lex_feats_nearwords_window2,train_pd.yelp_lex_feats_nearwords_window3,
	#							  train_pd.yelp_lex_feats_nearwords_window5,cat_vec))
	#test_feats =np.column_stack((testDataVecs,test_pd.yelp_lex_feats_nearwords_window2,test_pd.yelp_lex_feats_nearwords_window3,
	#							  test_pd.yelp_lex_feats_nearwords_window5,cat_vec_test))

	train_feats = []
	test_feats = []

	
	train_feats = np.column_stack((trainDataVecs,cat_vec,train_pd.token_distance_score))
	test_feats  = np.column_stack((testDataVecs,cat_vec_test,test_pd.token_distance_score))
		
	if algCD == 'Random Forest':
		test_pd = train_randomforest_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	elif algCD == 'Naive Bayes':
		test_pd = train_naiveBayes_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	elif algCD == 'SVM':
		test_pd = train_svc_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	elif algCD == 'ADA Boost':
		test_pd = train_adaboost_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)
	else:
		test_pd = train_xgboost_model_foreachPolarity(train_feats,test_feats,test_pd,polarities,train_pd)


	labels = []
	if 'is_neutral' in test_pd.columns:
		for ind,rev in test_pd.iterrows():
			labels.append(compute_label(rev['predicted_prob_positive'],rev['predicted_prob_negative'],rev['predicted_prob_neutral']))

	else:
		for ind,rev in test_pd.iterrows():
			labels.append(compute_label(rev['predicted_prob_positive'],rev['predicted_prob_negative']))


	test_pd['predictedLabels_multi'] = labels   

	acc = accuracy_score(test_pd['polarity'],test_pd['predictedLabels_multi'])
	fsco = f1_score(test_pd['polarity'],test_pd['predictedLabels_multi'],average='weighted')
	print(classification_report(test_pd['polarity'],test_pd['predictedLabels_multi'], target_names=test_pd['polarity'].unique()))

	print "--------------------------------------"
	print "\t Evaluation Results \t"
	print "--------------------------------------"

	print "Accuracy %f :"%acc
	print "F Score %f :"%fsco
	print ""
	return acc,fsco    
    
#main()
    