
# coding: utf-8

# In[33]:



# # Using opensource python libraries

# In[1]:

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import gensim
from gensim.models import Word2Vec
import nltk
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
import time
from nltk import tokenize
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()
import joblib

# In[2]:

warnings.filterwarnings("ignore")

current_directory = os.path.dirname(__file__)
parent_directory = os.path.split(current_directory)[0]
# In[3]:
def load_lexicon():
	lex = []
	fname = os.path.join(parent_directory,'lexica','pos_lexicon.txt')
	f = open(fname, "r")
	for line in f:
		tag = line.split()[0]
		lex.append(tag)
        
	return lex

yelp_lex = []
yelp_scores = []
def load_models(vec_fname,lex_filename):
	#LOADING MODELS
	#vectors_filename = "vectors_yelp_200.txt" #user provided
	vectors_filename = os.path.join(parent_directory,vec_fname)

	#lex_file = 'saif_lex/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt'
	lex_file = os.path.join(parent_directory,lex_filename)
	#lex_file = '../saif_lex/Amazon-laptops-electronics-reviews-AFFLEX-NEGLEX-unigrams.txt'
	   
	global pos_lexicon
	pos_lexicon = load_lexicon()
			
	global model
	model = gensim.models.Word2Vec.load_word2vec_format(vectors_filename,binary=False)
	global ndim
	ndim = model.vector_size
	filename = os.path.join(parent_directory, 'models', 'ote','otemodel.sav')
	global ssvm
	ssvm = pickle.load(open(filename, 'rb'))
	global index2word_set
	index2word_set = set(model.index2word)
	yelp_lex,yelp_scores = load_polarity_lexicon(lex_file)

#LOADING LEXICON FOR OTE TASK

    
#LOADING LEXICONS FOR POLARITY DETECTION TASK
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
# In[20]:

directory = os.path.join(parent_directory, 'models', 'acd')

categories = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.sav'):
                       s = str(file.lower())
                       result = re.search('model_(.*).sav', str(s))
                       categories.append(result.group(1))
models_acd = []
for c in categories:
          loaded_model = pickle.load(open(directory+'/model_'+ str(c)+'.sav', 'rb'))
          models_acd.append(loaded_model)

#POLARITY DETECTION MODELS
# Need to make it dynamic
models_pd = []
polarities = []
directory_pd = os.path.join(parent_directory, 'models', 'pd')
for root, dirs, files in os.walk(directory_pd):
    for file in files:
        if file.endswith('.sav'):
                       s = str(file.lower())
                       result = re.search('model_(.*).sav', str(s))
                       polarities.append(result.group(1))

for c in polarities:
          loaded_model = pickle.load(open(directory_pd+'/model_'+ str(c)+'.sav', 'rb'))
          models_pd.append(loaded_model)

#model_neutral = pickle.load(open('models/pd/model_neutral.sav', 'rb'))
#model_positive = pickle.load(open('models/pd/model_positive.sav', 'rb'))
#model_negative = pickle.load(open('models/pd/model_negative.sav', 'rb'))


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
       

# In[22]:
# need to make it dynamic

# In[4]:

stopwords =  nltk.corpus.stopwords.words('english')
not_remove = ['not','very','few','more','only','nor','but','don','t','and','with','no']
for word in not_remove:
    stopwords.remove(word)
    
#for polarity and category detection task
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

def review_to_wordlist( review, remove_stopwords=True ):
    # remove HTML
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    if remove_stopwords:
         words = [w for w in words if not w in stopwords]
            
    return ( words )


#for ote task
def review_to_words_ote( raw_review ):
    #remove non alphanumeric characters
    letters_only = re.sub("[^a-zA-Z0-9]", " ",raw_review) 
    # convert into lowercase and split text into words using split() function
    words = letters_only.split()
    # declaring empty array
    cleanwords = []
    for word in words: #if(word.lower() not in stopwords and len(word) > 2):
       
        cleanwords.append(word)
    return( " ".join( cleanwords ))



# In[5]:

# functions defined below are for ote feature extraction

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

# In[5]:

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


# In[6]:

def predict_aspect_terms(cleanText,test_sentences):
   
    predictedLabels = ssvm.predict(test_sentences)
    # aspect terms
    words = cleanText.split(" ")
    start_aspectTerm = False
    ote=''
    predictedLabels = predictedLabels[0]
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


# In[7]:

def predictOTE(text):
    #load model
    text = text.strip()
    #clean text
    cleanText = review_to_words_ote(text)
   
    #create vector features
    vec = create_vector_features(cleanText,model)
   
    
    # create vector features of next previous words
    #previous_vector_feats,next_vector_feats = create_next_prev_vector_features(cleanText,model,index2word_set)
    previous_vector_feats,next_vector_feats,second_next_vector_feats,second_previous_vector_feats = create_next_prev_vector_features(cleanText,model)
    #create morph features
    morph_feats = create_morph_feats(cleanText)
    
    # create POS features
    pos_sent_feats = create_pos_feats(cleanText,pos_lexicon)
   
    # create previous ,next word POS features
    #pos_sent_prev_feats,pos_sent_next_feats = create_prev_pos_feats(cleanText)
    pos_sent_prev_feats,pos_sent_next_feats,pos_sent_second_prev_feats,pos_sent_second_next_feats = create_prev_pos_feats(cleanText,pos_lexicon)
    	
    
    #prepare a array of features
    test_sentences = []
    x = cleanText.split(" ")
    test_word_feats = []
    for i,word in enumerate(x):
        test_word_features = []
        test_word_features.append(vec[i])
        test_word_features.append(morph_feats[i])
        test_word_features.append(pos_sent_feats[i])
        
        
        test_word_features.append(pos_sent_prev_feats[i])
        test_word_features.append(pos_sent_next_feats[i])
        test_word_features.append(pos_sent_second_prev_feats[i])
        test_word_features.append(pos_sent_second_next_feats[i])
        
        test_word_features.append(previous_vector_feats[i])
        test_word_features.append(next_vector_feats[i])
        test_word_features.append(second_previous_vector_feats[i])
        test_word_features.append(second_next_vector_feats[i])
        
        test_word_feats.append( np.concatenate((test_word_features[0],test_word_features[1],test_word_features[2],test_word_features[3],test_word_features[4],
                              test_word_features[5],test_word_features[6],test_word_features[7],test_word_features[8],test_word_features[9],test_word_features[10]),axis = 0))
        
   
    test_sentences_array = np.zeros((len(test_word_feats), len(test_word_feats[0])))
    index_i = 0
    for index_i in range(0,len(test_word_feats)):
        for index_j in range(0,len(test_word_feats[0])):
                test_sentences_array[index_i, index_j] = test_word_feats[index_i][index_j]

    test_sentences.append(test_sentences_array)
   
    aspect_term_predicted = predict_aspect_terms(cleanText,test_sentences)
    
    #post processing
    # assuming that  the length of aspect term <= 3
    rem_terms = []
    for term in aspect_term_predicted:
            
        if(len(term.split(' ')) % 2 == 0 and len(term.split(' ')) > 3 ):
            rem_terms.append(term) #terms to be removed later
            x = term.split(' ')
            for i in range(0,len(x),2):
                aspect_term_predicted.append(x[i] + ' ' + x[i+1])
                
    for term in rem_terms:
            aspect_term_predicted.remove(term)

    return aspect_term_predicted


# In[8]:

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
            reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features, index2word_set )
            counter = counter + 1.

    return reviewFeatureVecs


# In[26]:

def computeCategorySimilarity(x,model,index2word_set):
    
    # cat_words_vectors
    cat_words_vectors = []
    # Need to make it dynamic
    for cat in categories:
                similar_words_vec = []
                similar_words = model.most_similar(cat)
                for word in similar_words:
                    similar_words_vec.append(model[word[0]])
                cat_words_vectors.append(similar_words_vec)
                
                
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


# In[27]:
def return_prob_positive_class(ans):
    result = []
    for i in range(0,len(ans)):
        result.append(ans[i][1])
    return result
def predict_cat_labels(test_SF,testfeats):
    for c in range(0,len(categories)):
        colname = str(categories[c])
        test_SF[colname] = return_prob_positive_class(models_acd[c].predict_proba(testfeats))
    colnames = categories #['restaurant','price','misc','food','beverages','ambiance','service','location']
    row = test_SF.head(1)
    r_labels =[]
    for c in colnames:
            for index,row in test_SF.iterrows():
                if(row[c] >= 0.5):
                        r_labels.append(c)
        
    return r_labels

def predict_cat_labels_bulk(test_SF,testfeats):
    for c in range(0,len(categories)):
        colname = str(categories[c])
        test_SF[colname] = return_prob_positive_class(models_acd[c].predict_proba(testfeats))
   
    colnames = categories #['restaurant','price','misc','food','beverages','ambiance','service','location']
    f_pred = []
    for index,row in test_SF.iterrows():
        r_labels =[]
        for c in colnames:
                if(row[c] >= 0.5):
                        r_labels.append(c)
        f_pred.append(r_labels)
    return f_pred


# In[28]:

def return_category_vec(x):
    return model[x]


# In[35]:

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


# In[37]:

def compute_label(pos_prob,neg_prob,neu_prob):
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


# In[38]:

def predict_polarity(test_pd,testfeats):
    for p in range(0,len(polarities)):
        cname = polarities[p] + '_prob'
        test_pd[cname] = return_prob_positive_class(models_pd[p].predict_proba(testfeats))
   
    labels = []
    if('neutral_prob' in test_pd.columns):
            for index,rev in test_pd.iterrows():
                labels.append(compute_label(rev['positive_prob'],rev['negative_prob'],rev['neutral_prob']))
    else:
            for index,rev in test_pd.iterrows():
                labels.append(compute_label(rev['positive_prob'],rev['negative_prob'],0))

    test_pd['predictedLabels_multi'] = labels
    return test_pd
    
    


# In[39]:

def compute_polarity(polarity_sf):
    # category_vec feature
    cat_vec = []
    
    cat_vec_test = []
    for index,rev in polarity_sf.iterrows():
        cat_vec_test.append(return_category_vec(rev['modifiedcategory']))
  
    clean_test_reviews = []
    for review in polarity_sf['text']:
        clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, ndim, index2word_set )
    #find near words window 5,2
    
    near_words_5 = []
    near_words_2 = []
    for index,rev in polarity_sf.iterrows():
        near_words_5.append(find_near_words(rev['aspectterm'],rev['cleanText'],5))
        near_words_2.append(find_near_words(rev['aspectterm'],rev['cleanText'],2))
    
    polarity_sf['nearwords_window5'] = near_words_5
    polarity_sf['nearwords_window2'] = near_words_2
             
    
    
    #compute lexicon feats
    pos_feats1 = []
    pos_feats2 = []
   
    for index,rev in polarity_sf.iterrows():
        pos_feats1.append(create_lexicon_feats(rev['nearwords_window2'],yelp_lex,yelp_scores))
        pos_feats2.append(create_lexicon_feats(rev['nearwords_window5'],yelp_lex,yelp_scores))
        
    
    polarity_sf['yelp_lex_feats_nearwords_window2'] = pos_feats1
    polarity_sf['yelp_lex_feats_nearwords_window5'] = pos_feats2
  
    
    test_feats =np.column_stack((testDataVecs,polarity_sf.yelp_lex_feats_nearwords_window2,
                             polarity_sf.yelp_lex_feats_nearwords_window5,                           
                             cat_vec_test))

   
    del near_words_5,near_words_2,pos_feats1,pos_feats2
    predictions = predict_polarity(polarity_sf,test_feats)
    
    return predictions


# In[40]:

def assign_aspectterm_category(aspect_terms,categories_pred):
    #aspect terms : predicted from ote module
    #categories:predicted from aspect category detection system
    
    if len(aspect_terms) > 0:
            aspect_term_vec = []
            category_vec = []
            ans = []
            for cat in categories_pred:
                category_vec.append(return_category_vec(cat))


            for term in aspect_terms:
                similarity = []
                aspect_term_vec = []
                # dealing with multiword aspect term
                terms = term.split(' ')
                if len(terms) > 1:
                    multi_aspectterm_vec = []
                    for t in terms:
                        if t.lower() in index2word_set:
                            multi_aspectterm_vec.append(model[t.lower()])
                    if(len(multi_aspectterm_vec) > 0):        
                        for cat in category_vec:
                            temp = cosine_similarity(multi_aspectterm_vec,cat)
                            similarity.append(max(temp))

                # for single term aspect
                else:
                    if term.lower() in index2word_set:
                        aspect_term_vec.append(model[term.lower()])

                    if(len(aspect_term_vec) > 0):
                        for cat in category_vec:
                            similarity.append(cosine_similarity(aspect_term_vec,cat))

                if(len(similarity) > 0):
                    ans.append(categories_pred[np.argmax(similarity)])
                else:
                    ans.append([''])



            # ans are the assigned categories_pred from the predicted categories_pred
            # for those predicted categories_pred for which we have no aspect term assign 'null' as aspect term
            if(len(categories_pred) > 0):
                cats_with_no_aspect_terms = list(set(categories_pred) - set(ans))
                for cat in cats_with_no_aspect_terms:
                    aspect_terms.append('null')
                    ans.append(cat)
                return aspect_terms,ans
            
            else:
                return [],[]
    
    # case below deals for the situation when there is now aspect terms in the sentence
    else:
        aspect_terms = []
        for cat in categories_pred:
            aspect_terms.append('null')
        return aspect_terms,categories_pred    


# In[44]:

def predict_category(text):
    aspect_terms = predictOTE(text)
    ss = []
    ss.append(text)
    test_sf = pd.DataFrame({'text' : ss})
    test_sf['cleanText'] = test_sf['text'].apply(review_to_words)
    clean_test_reviews = []
    for review in test_sf['text']:
        clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, ndim, index2word_set )
    #test_sf['vectors'] = testDataVecs
    cs = np.zeros((len(test_sf),len(categories)),dtype="float32")
    i = 0
    for rev in test_sf['cleanText']:
        cs[i] =  computeCategorySimilarity(rev,model,index2word_set)
        i = i+1
    
    #test_sf['cs'] = cs
    test_feats = np.column_stack((testDataVecs,cs))
    predictedLabels = predict_cat_labels(test_sf,test_feats)
    aspect_t,cats = assign_aspectterm_category(aspect_terms,predictedLabels)
    
    # now predict sentiment of each category
    sent = []
    text_array = []
    cleanText_array = []
    for label in cats :
        #sent.append(test_sf['vectors'][0])
        text_array.append(test_sf['text'][0])
        cleanText_array.append(test_sf['cleanText'][0])
   
    polarity_sf = pd.DataFrame({'text' : text_array,'aspectterm' : aspect_t,'cleanText':cleanText_array,
                             'modifiedcategory' : cats})
    
    
    if(len(polarity_sf) > 0):
        polarity_sf = compute_polarity(polarity_sf)
        return polarity_sf[['text','aspectterm','modifiedcategory','predictedLabels_multi']]
        #return polarity_sf
    else:
        return []


# In[9]:

def predict_category_review(text):
    text = text.strip()
    text_array = tokenize.sent_tokenize(text)
    text = []
    cat = []
    at = []
    pol = []
    for tex in text_array:
        if(tex != ''):
                #aspect_term_predicted = predict_ote.predictOTE(tex)
                #cat_labels = PolarityDetection.predict_category(tex,aspect_term_predicted)
                cat_labels = predict_category(tex)
                if len(cat_labels) > 0:
                    for index,row in cat_labels.iterrows():
                        text.append(row['text'])
                        cat.append(row['modifiedcategory'])
                        at.append(row['aspectterm'])
                        pol.append(row['predictedLabels_multi'])
                    
    j = pd.DataFrame({'text' : text,'category' :cat,'aspectterm' :at,'polarity' :pol})
    
    #summary of overall review
    s = j[['category','polarity']]
    #summary of overall review
    s['count'] = s.groupby(['category','polarity']).polarity.transform('count')
    s = s.sort_values('polarity',ascending=False)
    idx = s.groupby(['category'])['count'].idxmax()
    s = s.loc[idx, ['category', 'polarity']]

    return j,s



# In[268]:

#bulk review analysis
def bulk_reviews(reviews):
    
    reviews_array = reviews.strip().split('\n')
    r_id = 0
    r_id_array = []
    sent = []
    for rev in reviews_array:
        if(rev != ''):
            sentences = nltk.sent_tokenize(rev)
            for s in sentences:
                r_id_array.append(r_id)
                sent.append(s)
            r_id = r_id + 1
    
    reviews = pd.DataFrame({'review_id' : r_id_array,'text' : sent})
    reviews['cleanText'] = reviews['text'].apply(review_to_words)
    ote = []
    for index,rev in reviews.iterrows():
        ote.append(predictOTE(rev['text']))
    reviews['aspect terms'] = ote
    clean_test_reviews = []
    for review in reviews['text']:
        clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ) )

    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, ndim, index2word_set )
    #reviews['vectors'] = testDataVecs
    cs = np.zeros((len(reviews),len(categories)),dtype="float32")
    i = 0
    for rev in reviews['cleanText']:
        cs[i] = computeCategorySimilarity(rev,model,index2word_set)
        i = i+1
    
    #reviews['cs'] = cs
    test_feats = np.column_stack((testDataVecs,cs))
   
    reviews['predictedLabels'] = predict_cat_labels_bulk(reviews,test_feats)
    
    ats = []
    cts = []
    pol = []
    for index,rev in reviews.iterrows():
        aspect_t,cats = assign_aspectterm_category(rev['aspect terms'],rev['predictedLabels'])
    
        # now predict sentiment of each category
        sent = []
        text_array = []
        cleanText_array = []
        for label in cats :
            #sent.append(rev['vectors'])
            text_array.append(rev['text'])
            cleanText_array.append(rev['cleanText'])
   
        polarity_sf = pd.DataFrame({'text' : text_array,'aspectterm' : aspect_t,'cleanText':cleanText_array,
                                 'modifiedcategory' : cats})
    
        if(len(polarity_sf) > 0):
            polarity_sf = compute_polarity(polarity_sf)
            pol.append(polarity_sf['predictedLabels_multi'])
            ats.append(aspect_t)
            cts.append(cats)
        else:
            pol.append([])
            ats.append([])
            cts.append([])
    reviews['category'] = cts
    reviews['aspect term'] = ats
    reviews['polarity'] = pol
    
    ats = []
    cts = []
    pol = []
    txt = []
    rid = []
    for index,rev in reviews.iterrows():
        for i in range(0,len(rev['aspect term'])):
            ats.append(rev['aspect term'][i])
            cts.append(rev['category'][i])
            pol.append(rev['polarity'][i])
            txt.append(rev['text'])
            rid.append(rev['review_id'])
            
    finalsf = pd.DataFrame({'review_id' : rid,'text':txt,'category' : cts,'aspect term' : ats,'polarity' : pol}) 
    
    #return category wise summary of sentiments
    s = finalsf[['review_id','category','polarity','aspect term']]
    
    s['count'] = s.groupby(['review_id','category','polarity']).polarity.transform('count')
    s = s.sort_values('polarity',ascending=False)
    idx = s.groupby(['review_id','category'])['count'].idxmax()
    s = s.loc[idx, ['review_id','category', 'polarity']]
    f = s.groupby(['category','polarity'])['polarity'].agg({'count':'count'})
    
    
    pos_c = []
    neg_c = []
    neu_c = []
    cate = f.index.levels[0]
    for c in cate:
            pos = 0
            neg= 0
            neu = 0
            for index,rows in f.iterrows():
                    if(index[0] == c and index[1] == 'positive'):
                        pos = rows['count']
                    if(index[0] == c and index[1] == 'negative'):
                        neg = rows['count']    
                    if(index[0] == c and index[1] == 'neutral'):
                        neu = rows['count'] 
            pos_c.append(pos)
            neg_c.append(neg)
            neu_c.append(neu)

    new_cat_wise = pd.DataFrame({'category' : cate,'sentiment.positive' : pos_c,'sentiment.negative' : neg_c,'sentiment.neutral' : neu_c})
    
    #producing aspect wise sentiment summary
    del s
    s = finalsf[['review_id','category','polarity','aspect term']]
    s['count'] = s.groupby(['review_id','aspect term','polarity']).polarity.transform('count')
    s = s.sort_values('polarity',ascending=False)
    
    idx = s.groupby(['review_id','aspect term'])['count'].idxmax()
    s = s.loc[idx, ['review_id','aspect term', 'polarity']]
    f = s.groupby(['aspect term','polarity'])['polarity'].agg({'count':'count'})
    
    
    pos_c = []
    neg_c = []
    neu_c = []
    cate = f.index.levels[0]
    for c in cate:
            pos = 0
            neg= 0
            neu = 0
            for index,rows in f.iterrows():
                    if(index[0] == c and index[1] == 'positive'):
                        pos = rows['count']
                    if(index[0] == c and index[1] == 'negative'):
                        neg = rows['count']    
                    if(index[0] == c and index[1] == 'neutral'):
                        neu = rows['count'] 
            pos_c.append(pos)
            neg_c.append(neg)
            neu_c.append(neu)

    new_term_wise = pd.DataFrame({'aspect term' : cate,'sentiment.positive' : pos_c,'sentiment.negative' : neg_c,'sentiment.neutral' : neu_c})
    new_term_wise['count'] = new_term_wise['sentiment.positive'] + new_term_wise['sentiment.neutral'] + new_term_wise['sentiment.negative']
    new_term_wise = new_term_wise[new_term_wise['aspect term'] != 'null']
    
    return new_cat_wise,new_term_wise



# In[ ]:



