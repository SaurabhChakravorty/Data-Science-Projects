# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


os.chdir("D:/Dropbox/Lehre/Digital Analytics/Excercises/Classification")
data_ori = pd.read_csv('YoungBank.csv')
print(data_ori.shape)
# types
print(data_ori.dtypes)
# feature names
print(list(data_ori))
# head
print(data_ori.head(6))
# descriptions, change precision to 2 places
print(data_ori.describe())

# Downsampling
print(data_ori['Personal_Loan'].value_counts())
# Separate majority and minority classes
data_ori_majority = data_ori[data_ori.Personal_Loan=='no']
data_ori_minority = data_ori[data_ori.Personal_Loan=='yes']
 
# Downsample majority class
from sklearn.utils import resample
data_ori_majority_downsampled = resample(data_ori_majority, 
                                         replace=False,    # sample without replacement
                                         n_samples=len(data_ori_minority),     # to match minority class
                                         random_state=0) # reproducible results
 # Combine minority class with downsampled majority class
data_ori_downsampled = pd.concat([data_ori_majority_downsampled, data_ori_minority])
print(data_ori_downsampled['Personal_Loan'].value_counts())

# Create X and Y
X_ori = data_ori_downsampled.drop('Personal_Loan', axis = 1)
Y = data_ori_downsampled['Personal_Loan']

# Normalize X
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
print(X.head(6))

# Partition into Training and Test sample
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.5, random_state=0)
print(Y_train.value_counts())
print(Y_test.value_counts())


#create report dataframe
report = pd.DataFrame(columns=['Model','Acc.Train','Acc.Test'])


################
#     KNN      #
################

from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=7)
knnmodel.fit(X_train, Y_train)
Y_train_pred = knnmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = knnmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#find optimal k
accuracies = []
for k in range(1, 21):
    knnmodel = KNeighborsClassifier(n_neighbors=k)
    knnmodel.fit(X_train, Y_train)
    Y_test_pred = knnmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    print(k, accte)
    accuracies.append(accte)
plt.plot(range(1, 21), accuracies)
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies')
plt.show()
opt_k = np.argmax(accuracies) + 1
print('Optical k =', opt_k)

knnmodel = KNeighborsClassifier(n_neighbors=opt_k)
knnmodel.fit(X_train, Y_train)
Y_train_pred = knnmodel.predict(X_train)
acctr = accuracy_score(Y_train, Y_train_pred)
Y_test_pred = knnmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['k-NN', acctr, accte]

