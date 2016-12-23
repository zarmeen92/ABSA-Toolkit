NRC Amazon Laptop Sentiment Lexicons
Version 1.0
26 September 2014
Copyright (C) 2014 National Research Council Canada (NRC)
Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)

**********************************************
TERMS OF USE
**********************************************

1. This lexicon can be used freely for research purposes. 
2. The papers listed below provide details of the creation and use of 
   the lexicon. If you use a lexicon, then please cite the associated 
   paper:
	Kiritchenko, S., Zhu, X., Cherry, C., and Mohammad, S. (2014) 
	NRC-Canada-2014: Detecting Aspects and Sentiment in Customer 
	Reviews. Proceedings of the 8th International Workshop on 
	Semantic Evaluation Exercises (SemEval-2014), Dublin, Ireland, 
	2014.

3. If interested in commercial use of the lexicon, send email to the 
   contact. 
4. If you use the lexicon in a product or application, then please 
   credit the authors and NRC appropriately. Also, if you send us an 
   email, we will be thrilled to know about how you have used the 
   lexicon.
5. National Research Council Canada (NRC) disclaims any responsibility 
   for the use of the lexicon and does not provide technical support. 
   However, the contact listed above will be happy to respond to 
   queries and clarifications.
6. Rather than redistributing the data, please direct interested 
   parties to this page:
   http://www.purl.com/net/lexicons 

Please feel free to send us an email:
- with feedback regarding the lexicon. 
- with information on how you have used the lexicon. 
- if interested in having us analyze your data for sentiment, emotion, 
  and other affectual information.
- if interested in a collaborative research project.

**********************************************
DATA SOURCE
**********************************************

The NRC Amazon Laptop Sentiment Lexicons are automatically generated from the following data source:
customer reviews on electronic products that mention 'laptop' or 'notebook'; the reviews were collected from Amazon.com by McAuley and Leskovec (Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings of the 7th ACM conference on Recommender systems, pages 165–172.)


**********************************************
FILE FORMAT
**********************************************

Each line in the lexicons has the following format:
<term><tab><score><tab><Npos><tab><Nneg>

<term> can be a unigram or a bigram;
<score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
<Npos> is the number of times the term appears in the positive class, ie. in tweets with positive hashtag or emoticon;
<Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative hashtag or emoticon.


**********************************************
AffLex and NegLex
**********************************************

Both parts, AffLex and NegLex, of each lexicon are contained in the same file. The NegLex entries have suffixes '_NEG' or '_NEGFIRST'.

In the unigram lexicon:
'_NEGFIRST' is attached to terms that directly follow a negator;
'_NEG' is attached to all other terms in negated contexts (not directly following a negator).

In the bigram lexicon:
'_NEG' is attached to all terms in negated contexts.

Both suffixes are attached only to nouns, verbs, adjectives, and adverbs. All other parts of speech do not get these suffixes attached. 


**********************************************
More Information
**********************************************
Details on the process of creating the lexicons can be found in:
Kiritchenko, S., Zhu, X., Mohammad, S. (2014). Sentiment Analysis of Short Informal Texts.  Journal of Artificial Intelligence Research, 50:723-762, 2014.

 
