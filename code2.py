# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:17:57 2021

@author: varun
"""

# initialize afinn sentiment analyzer
from afinn import Afinn
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix



af = Afinn()

def scoring(text):
    try:
        x=af.score(text)
    except:
        x=0

    return(x)        
processed_tweets['liststring']=None
processed_tweets['liststring'] = [','.join(map(str, l)) for l in processed_tweets['text']]
processed_tweets['liststring']=processed_tweets['liststring'].replace(to_replace=r',', value=' ', regex=True)

processed_tweets['score']=processed_tweets.apply(lambda row : scoring(row['liststring']), axis = 1) 

processed_tweets.loc[processed_tweets['score']>0,'sentiment_new']=1
processed_tweets.loc[processed_tweets['score']<0,'sentiment_new']=-1
processed_tweets.loc[processed_tweets['score']==0,'sentiment_new']=0    
  
accuracy_score(list(processed_tweets['sentiment']),list(processed_tweets['sentiment_new']))
#processed_tweets.head()
print(confusion_matrix(list(processed_tweets['sentiment_new']),list(processed_tweets['sentiment'])))





average_precision = average_precision_score(list(processed_tweets['sentiment_new']),list(processed_tweets['sentiment']))

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))



from sklearn import metrics

# Constants confusion matrix
print(metrics.confusion_matrix(y_true, y_pred))

# Print the precision and recall, among other metrics
print(metrics.classification_report(list(processed_tweets['sentiment']),list(processed_tweets['sentiment_new']), digits=3))