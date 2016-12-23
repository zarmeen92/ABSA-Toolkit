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

**2. flaskWebApp:** contains the code for generating Flask Web Application. You don't need to modify this code.

**3. lexicons:** contains polarity lexicons files for computing polarity scores. You can place lexicon file of your choice inside this folder. Currently this folder contains lexicons publicly available at http://saifmohammad.com/WebPages/lexicons.html
For any domain, you can use  wnscores_inquirer.txt lexicon (Source: http://compprag.christopherpotts.net/iqap-experiments.html)

**4. models:** All the trained models saved during training phase are saved in this folder. Inside /acd, models trained for each aspect category is saved. Inside /ote, model trained for aspect term detection is saved. Inside /pd, models trained for polarity detection are saved.

**Note : Before running script for training, remove all files inside models/acd, models/pd, models/ote folders**

**5. wordembeddings:** contains word vectors in txt format. Currently, we have amazon200.txt suitable for Electronic Products dataset. vector_yelp_200.txt suitable for Restaurant domain. For any other domain, you can either train your own Word2Vec model and save word embeddings in txt format or you can use Glove pretrained models available at http://nlp.stanford.edu/projects/glove/

## Training Models for Aspect-Based Sentiment Analysis
Follow instructions written below:
1. Clone/Download repository
2. Install all the required python libraries mentioned above
3. Place your csv files for training and testing inside /data folder. Currently, data folder contains Restaurant and Laptop datasets 
4. Place word embedding file inside /wordembeddings folder. You can use already placed wordembeddings file if the dataset is of Restaurant or Electronic Product domain. Else you can download Glove pretrained vectors

5. Make sure to remove all files inside models/acd, models/pd, models/ote folders before training
6. Run script absa.py for training aspect-based sentiment analysis models using command below
        **python absa.py -trainfile -testfile -vectors -lexicon**
   Try running,
   python absa.py data/restaurants/train.csv data/restaurants/test.csv vectors_yelp_200.txt lexicons/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt 

  for restaurant domain application

Once the models are trained, you will see summary as shown below
![Summary](https://github.com/zarmeen92/ABSA-Toolkit/blob/master/absa-snapshots/Capture6.PNG)


After training phase is complete you are ready to use web application

## Production Phase for Aspect-Based Sentiment Analysis
**python absaweb.py -vectors -lexicon**

Use the same vector file and lexicon file as used in training phase
For example,

** python absaweb.py vectors_yelp_200.txt lexicons/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt**
This will start Flash Application accessible at 127.0.0.1:9000 on your browser
### Snapshots 

![Summary](https://github.com/zarmeen92/ABSA-Toolkit/blob/master/absa-snapshots/Capture7.PNG)
![Summary](https://github.com/zarmeen92/ABSA-Toolkit/blob/master/absa-snapshots/Capture8.PNG)
![Summary](https://github.com/zarmeen92/ABSA-Toolkit/blob/master/absa-snapshots/Capture9.PNG)
![Summary](https://github.com/zarmeen92/ABSA-Toolkit/blob/master/absa-snapshots/Capture10.PNG)













        - 


