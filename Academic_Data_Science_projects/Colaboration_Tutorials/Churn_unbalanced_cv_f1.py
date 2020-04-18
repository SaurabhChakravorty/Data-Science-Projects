# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


os.chdir("D:/Dropbox/Lehre/Digital Analytics/Excercises/Classification")
#data_ori = pd.read_csv('churn_balanced.csv')
data_ori = pd.read_excel('churn_unbalanced.xls')
print(data_ori.shape)

X_ori = data_ori.drop('CHURN', axis = 1)
Y = data_ori['CHURN']

# standardize data = (data_ori-data_ori.mean())/data_ori.std()
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
print(X.head(6))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)  #random_state=0
print(Y_train.value_counts())
print(Y_test.value_counts())

#create report dataframe
report = pd.DataFrame(columns=['Model','Mean Training','Standard Deviation','Test'])


#######################
# Logistic Regression #
#######################

### unbalanced ###

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(lrmodel, X_train, Y_train, scoring='accuracy', cv = 10)
print("Accuracies = ", accuracies)
print("Mean = ", accuracies.mean())
print("SD = ", accuracies.std())
lrmodel.fit(X_train, Y_train)
Y_test_pred = lrmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['LR unbalanced Acc', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])

# using f1 score
from sklearn.linear_model import LogisticRegression
lrmodel_f1 = LogisticRegression()
from sklearn.model_selection import cross_val_score
f1es = cross_val_score(lrmodel_f1, X_train, Y_train, scoring='f1', cv = 10)
print("f1es = ", f1es)
print("Mean = ", f1es.mean())
print("SD = ", f1es.std())
lrmodel_f1.fit(X_train, Y_train)
Y_test_pred = lrmodel_f1.predict(X_test)
cmte_f1 = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte_f1)
f1te = f1_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['LR unbalanced f1', f1es.mean(), f1es.std(), f1te]
print(report.loc[len(report)-1])

### balanced ###

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression(class_weight='balanced')
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(lrmodel, X_train, Y_train, scoring='accuracy', cv = 10)
print("Accuracies = ", accuracies)
print("Mean = ", accuracies.mean())
print("SD = ", accuracies.std())
lrmodel.fit(X_train, Y_train)
Y_test_pred = lrmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['LR balanced Acc', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])

# using f1 score
from sklearn.linear_model import LogisticRegression
lrmodel_f1 = LogisticRegression(class_weight='balanced')
from sklearn.model_selection import cross_val_score
f1es = cross_val_score(lrmodel_f1, X_train, Y_train, scoring='f1', cv = 10)
print("f1es = ", f1es)
print("Mean = ", f1es.mean())
print("SD = ", f1es.std())
lrmodel_f1.fit(X_train, Y_train)
Y_test_pred = lrmodel_f1.predict(X_test)
cmte_f1 = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte_f1)
f1te = f1_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['LR balanced f1', f1es.mean(), f1es.std(), f1te]
print(report.loc[len(report)-1])


#################
# Random Forest #
#################

### unbalanced ###

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 4.,  5.,  6.,  7.,  8.],
    'n_estimators': [ 10,  50,  100, 150, 200]
}
CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, scoring='accuracy', cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)
#use the best parameters
rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
rfmodel.fit(X_train, Y_train)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
Y_test_pred = rfmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['RF unbalanced Acc', 
                          CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], 
                          CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_], accte]
print(report.loc[len(report)-1])


from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 4.,  5.,  6.,  7.,  8.],
    'n_estimators': [ 10,  50,  100, 150, 200]
}
CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, scoring='f1', cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)
#use the best parameters
rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
rfmodel.fit(X_train, Y_train)
Y_test_pred = rfmodel.predict(X_test)
cmte_f1 = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte_f1)
f1te = f1_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['RF unbalanced f1', 
                          CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], 
                          CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_], f1te]
print(report.loc[len(report)-1])

### balanced ###

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(class_weight='balanced', random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 4.,  5.,  6.,  7.,  8.],
    'n_estimators': [ 10,  50,  100, 150, 200]
}
CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, scoring='accuracy', cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)
#use the best parameters
rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
rfmodel.fit(X_train, Y_train)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
Y_test_pred = rfmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['RF balanced Acc', 
                          CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], 
                          CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_], accte]
print(report.loc[len(report)-1])


from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(class_weight='balanced', random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 4.,  5.,  6.,  7.,  8.],
    'n_estimators': [ 10,  50,  100, 150, 200]
}
CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, scoring='f1', cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)
#use the best parameters
rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
rfmodel.fit(X_train, Y_train)
Y_test_pred = rfmodel.predict(X_test)
cmte_f1 = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte_f1)
f1te = f1_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['RF balanced f1', 
                          CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], 
                          CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_], f1te]
print(report.loc[len(report)-1])


################
# Final Report #
################

print(report)

