import os
import numpy as np
import pandas as pd
import nltk
from sklearn import metrics
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
import datetime
import datetime
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import networkx as nx
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from calendar import monthrange
import time
import timeit
import sklearn
from sklearn.neighbors import DistanceMetric
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC 
import sklearn.metrics as metrics
import nltk

def process(test, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    clean = re.compile('<.*?>')
    test=re.sub(clean, '', test)
    posMapping = {# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
            "N":'n',
            "V":'v',
            "J":'a',
            "R":'r'}
    url_reg='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
    words=re.sub(url_reg, '', test)
    words=words.replace('\'s', '')
    words=words.replace('\'', '')
    words = [word.lower() for word in words]  
    listnew=[]
    for i in words:
          if i in string.punctuation:
              listnew.append(" ")
          else:
              listnew.append(i)        
    words="".join(listnew)
    words=nltk.word_tokenize(words)
    def lemmatizerfunction(word,pos):
          if pos==None:
              return(None)
          else:
              result=lemmatizer.lemmatize(word,pos)
              return(result)  
    
    list1=nltk.pos_tag(words)
    
    list1=[ele[1] for ele in list1]   
    list1=[ele[0] for ele in list1]     
    def converter(x):
        if x in posMapping:
            return(posMapping[x])
        else:     
            return('n')      
    list1=[converter(i) for i in list1]  
    listfinal=list(zip(words,list1))
    listfinal=[lemmatizerfunction(i[0],i[1]) for i in listfinal]
    return(listfinal) 

def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    df['text']=df.apply(lambda row : process(row['text'],lemmatizer), axis = 1) 
    return(df)

def create_features(processed_tweets, stopwords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords,min_df=2,ngram_range=(1,2))
    corpus=[" ".join(review) for review in processed_tweets['text'].values]
    sm=vectorizer.fit_transform(corpus)
    return(vectorizer,sm)

def create_labels(processed_tweets):
    df=processed_tweets
    Y=df['sentiment']
    return(Y)


# execute code
def evaluate_classifier(classifier, X_validation, y_validation):
    y_validation=pd.to_numeric(y_validation)
    predictions=classifier.predict(X_validation)
    
    actuals=y_validation
    Z=list(zip(actuals,predictions))
    
    from sklearn.metrics import accuracy_score
    #print(predictions,actuals)
    
    # Print the precision and recall, among other metrics
    #print(metrics.classification_report(predictions,actuals))
    
    return(accuracy_score(predictions,actuals))

path=r'E:\CS-412-IML\SentimentAnalysis-main'
os.chdir(path)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
filename='train.csv'
lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')

df=pd.read_csv(filename)
df.head()

df.columns=['ID','text','selected_text','sentiment']
df=df[['text','sentiment']]
df=df.replace({'positive': 1,'negative':-1,'neutral':0})
df=df.loc[df['sentiment'].isin([1,-1,0])]
df.head()

df=df.reset_index() 
df=df.dropna(subset=['text', 'sentiment'])
processed_tweets = process_all(df)
processed_tweets.head()

(tfidf, X) = create_features(processed_tweets, stopwords)

y = create_labels(processed_tweets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
y_train=y_train.astype('int')

# -------------Grid search to parameter optimization------------------------
from sklearn.model_selection import GridSearchCV
param_grid=[{
        'max_depth':[1,2,5,7,10,100,200,300,500],
        'criterion':['gini','entropy'], 'splitter':['best','random'],
        'random_state':[0]}]


print("strting grid search")  
decisionTree_cv=GridSearchCV( DecisionTreeClassifier(), param_grid, verbose = 3, cv=3, n_jobs = 16)
decisionTree_cv.fit(X_train,y_train)
print("tuned hpyerparameters :(best parameters) ",decisionTree_cv.best_params_)
print("accuracy :",decisionTree_cv.best_score_)

best_params = decisionTree_cv.best_params_

decTree2=DecisionTreeClassifier(max_depth = best_params['max_depth'],criterion = best_params['criterion'],splitter = best_params['splitter'],random_state = 0)
decTree2.fit(X_train,y_train)
pred = decTree2.predict(X_test)
print(pred) 
print("score",decTree2.score(X_test,y_test))
print(metrics.classification_report(pred,y_test))

#-------------Accuracy metrics generation------------------:

def best_model_selection_report():
    mainlist=[]

    for parameters in param_grid:
        for splitt in parameters['splitter']:
            for criteria in parameters['criterion']:
                for maxdepth in parameters['max_depth']:
                    dectree_clfr = DecisionTreeClassifier(max_depth = maxdepth,criterion = criteria, splitter = splitt, random_state = 0)
                    dectree_clfr.fit(X_train,y_train)
                    accuracy = evaluate_classifier(dectree_clfr, X_test, y_test)
                    mainlist.append((criteria,maxdepth,splitt,accuracy))
    print(mainlist)
    return mainlist      

report = best_model_selection_report()
print(report)
results=pd.DataFrame(report)
results.columns=['Criterion','Max_depth','Split','accuracy']
results= results.groupby(['Criterion','Max_depth','Split']).agg({'accuracy': ['mean', 'min', 'max']})
results.columns = ['accuracy_mean', 'accuracy_min', 'accuracy_max']
results=pd.DataFrame(results)
results=results.reset_index()
results.to_excel(r'E:\CS-412-IML\SentimentAnalysis-main\DecisionTree_report.xlsx', index = False)
print(results)
