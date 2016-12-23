# ABSA-Toolkit
ABSA (Aspect-Based Sentiment Analysis) Toolkit developed for performing aspect-level sentiment analysis on customer reviews. The system has two main phases, a development phase and a production phase. Development Phase allows user to train models for performing aspect level sentiment analysis tasks on target domain. In the production phase, a web application is generated through which end user can submit reviews to analyze aspect level sentiments.

The system is developed using Python 2.7

## Python Libraries
Following libraries are required to run python scripts
- pandas
- numpy
- sklearn
- gensim
- sys
- warnings
- flask
- json
- time
- pickle
- nltk
- xgboost
- pystruct
- re

Download the above mentioned libraries using  command **pip install package-name**
To run xgboost on Windows, please refer to installation guide available here : https://github.com/dmlc/xgboost/blob/master/doc/build.md
After installing nltk, run following commands in python environment : 
import nltk
nltk.download()

To install nltk tagger and stopwords list
## Code Organization
**1. data:** contains training and testing datasets. Place your training and testing datasets inside data folder. Our system expects input datasets to be in specific format. Please see data/restaurants/train.csv for input data format.

**2. 


