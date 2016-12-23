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

**2. flaskWebApp: ** contains the code for generating Flask Web Application. You don't need to modify this code.

**3. lexicons: ** contains polarity lexicons files for computing polarity scores. You can place lexicon file of your choice inside this folder. Currently this folder contains lexicons publicly available at http://saifmohammad.com/WebPages/lexicons.html
For any domain, you can use  wnscores_inquirer.txt lexicon (Source: http://compprag.christopherpotts.net/iqap-experiments.html)

**4. models: ** All the trained models saved during training phase are saved in this folder. Inside /acd, models trained for each aspect category is saved. Inside /ote, model trained for aspect term detection is saved. Inside /pd, models trained for polarity detection are saved.

**Note : Before running script for training, remove all files inside models/acd, models/pd, models/ote folders**

**5. wordembeddings:** contains word vectors in txt format. Currently, we have amazon200.txt suitable for Electronic Products dataset. vector_yelp_200.txt suitable for Restaurant domain. For any other domain, you can either train your own Word2Vec model and save word embeddings in txt format or you can use Glove pretrained models available at http://nlp.stanford.edu/projects/glove/






        - 


