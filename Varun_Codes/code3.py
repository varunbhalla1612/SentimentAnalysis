# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:34:53 2020

@author: Varun
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:02:26 2020

@author: Varun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:34:53 2020

@author: Varun
"""

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
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC 
import sklearn.metrics as metrics
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


############################################################3
path=r'C:\Users\varun\OneDrive\Desktop\Files\IML\Course_Project\Dataset'
os.chdir(path)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
filename='train.csv'
lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')

############################################################3
df=pd.read_csv(filename)
df.head()
#df=df[2:]
df.columns=['ID','text','selected_text','sentiment']
df=df[['text','sentiment']]

df=df.replace({'positive': 1,'negative':-1,'neutral':0})
df=df.loc[df['sentiment'].isin([1,-1,0])]
############################################################3
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
    print(predictions,actuals)
    
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(predictions,actuals))
    
    return(accuracy_score(predictions,actuals))


df=df.reset_index() 
df=df.dropna(subset=['text', 'sentiment'])
processed_tweets = process_all(df)

# execute this code 
(tfidf, X) = create_features(processed_tweets, stopwords)

y = create_labels(processed_tweets)

##############withoutoversampling###########################3
kernel='linear'
C=10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
classifier = learn_classifier(X_train, y_train, kernel,C,1)
y_test=y_test.astype('int')
accuracy = evaluate_classifier(classifier, X_test, y_test)
print(accuracy) 



###############OVersampling#######################
kernel='rbf'
C=10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
y_train=y_train.astype('int')
oversample = SMOTE()
X_over, y_over = oversample.fit_resample(X_train, y_train)
classifier = learn_classifier(X_over, y_over, kernel,C,1)
y_test=y_test.astype('int')
accuracy = evaluate_classifier(classifier, X_test, y_test)
print(accuracy) 



###################################################################
import sklearn import model_selection
import sklearn.model_selection

import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, random_state=1, shuffle=True)
kf
#wordcloud


def learn_classifier(X_train, y_train, kernel,C):
    clf = SVC(kernel=kernel, C=C)
    #print(kernel)
    y_train=y_train.astype('int')
    clf.fit(X_train,y_train)
    return(clf)


def best_model_selection(kf, X, y):
    mainlist=[]
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
            list1=[]
            for train_index, test_index in kf.split(X):
                for C in [0.1,1,10,100,1000,10000]:
                     #print(kernel)
                     X_train, X_test = X[train_index], X[test_index]
                     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                     y_train=y_train.astype('int')            
                     #oversample = SMOTE()
                     #X_over, y_over = oversample.fit_resample(X_train, y_train)
                     classifier_new = learn_classifier(X_train, y_train, kernel, C)
                     accuracy = evaluate_classifier(classifier_new, X_test, y_test)
                     #list1.append(accuracy)
                     print(accuracy)
                     mainlist.append((kernel,C,accuracy))

    return mainlist

best_kernel = best_model_selection(kf, X, y)
results=pd.DataFrame(best_kernel)
results.columns=['kernel','C','accuracy']
results= results.groupby(['kernel', 'C']).agg({'accuracy': ['mean', 'min', 'max']})
results.columns = ['accuracy_mean', 'accuracy_min', 'accuracy_max']
results=pd.DataFrame(results)
results=results.reset_index()

##################grid search#####################
from sklearn.model_selection import GridSearchCV 
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_over, y_over) 
print(grid.best_params_) 

                     
#########################################################################################3
##############testing####################################
df=pd.read_excel(filename,sheet_name='Obama')
df.head()
df=df[2:]
df.columns=['X','date','time','tweet','class','Y','Z']
df=df[['tweet','class']]
df=df.loc[df['class'].isin([1,-1,0])]
df=df.dropna(subset=['tweet','class'])
y=df['class']
df=df[['tweet']]
df=df.reset_index() 
unlabeled_tweets = process_all(df)
corpus=[" ".join(review) for review in unlabeled_tweets ['tweet'].values]
New=tfidf.transform(corpus)
results=classifier.predict(New)
print(metrics.classification_report(results,list(y)))


