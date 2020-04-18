# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 03:09:00 2019

@author: Saurabh
"""

import pandas as pd
import numpy as np
df = pd.read_csv("D:\\MADS_Subjects\\IntroductionOfDataAnalytics\\Data_Files\\YoungBank.csv")

#Gives details of all the parameters of df columns
df.describe()

df = df[df.Experience != -3].reset_index(drop=True)

from sklearn import preprocessing
sscaler = preprocessing.StandardScaler()
data_standardized = df
data_standardized.iloc[:,0:7] = sscaler.fit_transform(data_standardized.iloc[:,0:7])



import seaborn as sns
sns.boxplot(data=df, orient="h", palette="Set2")



from sklearn.model_selection import train_test_split
from sklearn import metrics
X = df.drop('Personal_Loan',axis = 1)
Y = df.Personal_Loan
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.4, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
# try K=1 through K=25 and record testing accuracy
k_range = range(1, 15)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test, y_pred))
    
    
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')