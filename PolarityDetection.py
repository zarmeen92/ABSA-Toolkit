
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import gensim
from gensim.models import Word2Vec
import numpy as np
import re
import os
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pickle

# In[29]:

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



# In[41]:

def create_lexicon_feats(x,yelp_lex,yelp_scores):
    x = x.lower()
    x = re.sub('\.',' ',x).strip()
    words = x.split(' ')
    count = 0
    for j in range(0,len(words)):
        if (words[j] not in ['don\'t','no','dont','not','never']):
            for i in range(0,len(yelp_lex)):
                
                 if words[j] == yelp_lex[i]:
                    if(j >0):
                        if(words[j-1] in ['don\'t','no','dont','not','never']):
                            count = count + (-1)*float(yelp_scores[i])
                        else:
                            count = count + float(yelp_scores[i])
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
            'subsample':[i/100.0 for i in range(60,90,5)],
            'min_child_weight' :range(1,6,2),
            'colsample_bytree' :[i/100.0 for i in range(60,90,5)],
            'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
            'learning_rate':[0.1],
            'objective':['binary:logistic'],
            'scale_pos_weight' : [1,5,10],
            'seed':[5]
           }
    xgb_model = xgboost.XGBClassifier()
    for pol in polarities:
				print ""
				print "----------------------------------------"
				print "Training Classifier for %s Class " %pol
				print "----------------------------------------"
				print ""
				clf = RandomizedSearchCV(xgb_model, x_params,cv = 3,scoring=f1_scorer)
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
    if(pos_prob < 0.2 and neu_prob < 0.2 and neg_prob < 0.2):
        return "positive"
    
    elif(abs(neu_prob - neg_prob) < 0.2 and pos_prob < 0.2):
        return "neutral"
    
    elif(abs(neu_prob - pos_prob) < 0.2 and neg_prob < 0.2):
        return "neutral"
    
    elif(neu_prob > pos_prob and neu_prob > neg_prob):
        return "neutral"
    
    elif( pos_prob > neu_prob  and abs(neg_prob - pos_prob) > 0.2 and pos_prob > neg_prob):
        return "positive"
    elif(neg_prob > pos_prob  and abs(neg_prob - pos_prob) > 0.2 and neg_prob > neu_prob):
        return "negative"
    else:
        return "positive"

    

def main(train_pd,test_pd,model,lex_file):
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

    print "Extracting features..."
    near_words = []
    for ind,rev in train_pd.iterrows():
        near_words.append(find_near_words(rev['aspect term'],rev['cleanText'],5))

    train_pd['nearwords_window5'] = near_words
    near_words = []
    for ind,rev in test_pd.iterrows():
        near_words.append(find_near_words(rev['aspect term'],rev['cleanText'],5))
    test_pd['nearwords_window5'] = near_words
    near_words = []
    for ind,rev in train_pd.iterrows():
        near_words.append(find_near_words(rev['aspect term'],rev['cleanText'],2))
    train_pd['nearwords_window2'] = near_words
    near_words = []
    for ind,rev in test_pd.iterrows():
        near_words.append(find_near_words(rev['aspect term'],rev['cleanText'],2))
    test_pd['nearwords_window2'] = near_words


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

    print "Extracting Lexicon Features..."
    train_pd['yelp_lex_feats_nearwords_window2'] = create_lexicon_features_df(train_pd,'2',yelp_lex,yelp_scores)
    test_pd['yelp_lex_feats_nearwords_window2'] = create_lexicon_features_df(test_pd,'2',yelp_lex,yelp_scores)
    train_pd['yelp_lex_feats_nearwords_window5'] = create_lexicon_features_df(train_pd,'5',yelp_lex,yelp_scores)
    test_pd['yelp_lex_feats_nearwords_window5'] = create_lexicon_features_df(test_pd,'5',yelp_lex,yelp_scores)

    # ## Create Binary Columns for Polarity Classes

    # In[75]:

    polarities = set(train_pd['polarity'])
    print "Creating binary Columns for each polarity...."
    train_pd = create_binary_columns(train_pd,polarities)
    test_pd = create_binary_columns(test_pd,polarities)

    train_feats =np.column_stack((trainDataVecs,train_pd.yelp_lex_feats_nearwords_window2,
                                  train_pd.yelp_lex_feats_nearwords_window5,cat_vec))
    test_feats =np.column_stack((testDataVecs,test_pd.yelp_lex_feats_nearwords_window2,
                                  test_pd.yelp_lex_feats_nearwords_window5,cat_vec_test))

    print "------- Training Sentiment Polarity Detection Classifiers --------"
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
    fsco = f1_score(test_pd['polarity'],test_pd['predictedLabels_multi'],average='macro')

    print "--------------------------------------"
    print "\t Evaluation Results \t"
    print "--------------------------------------"

    print "Accuracy %f :"%acc
    print "F Score %f :"%fsco
    print ""
    return acc    
    
#main()
    