# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score


os.chdir("D:/Dropbox/Lehre/Digital Analytics/Excercises/Classification")
data_ori = pd.read_csv('churn_balanced.csv')
print(data_ori.shape)
# types
print(data_ori.dtypes)
# feature names
print(list(data_ori))
# head
print(data_ori.head(6))
# descriptions, change precision to 2 places
print(data_ori.describe())

X_ori = data_ori.drop('CHURN', axis = 1)
Y = data_ori['CHURN']

# standardize data = (data_ori-data_ori.mean())/data_ori.std()
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
print(X.head(6))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print(Y_train.value_counts())
print(Y_test.value_counts())

#create report dataframe
report = pd.DataFrame(columns=['Model','Mean Acc. Training','Standard Deviation','Acc. Test'])


#######################
#   Defining Models   #
#######################

from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=7)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(knnmodel, X_train, Y_train, scoring='accuracy', cv=10)
knnmodel.fit(X_train, Y_train)
Y_test_pred = knnmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['k-NN', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])

from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
accuracies = cross_val_score(nbmodel, X_train, Y_train, scoring='accuracy', cv=10)
nbmodel.fit(X_train, Y_train)
Y_test_pred = nbmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Naive Bayes', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
accuracies = cross_val_score(rfmodel, X_train, Y_train, scoring='accuracy', cv=10)
rfmodel.fit(X_train, Y_train)
Y_test_pred = rfmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Random Forest', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])


###################################
# Comparing with Voting Ensembles #
###################################

from mlxtend.classifier import EnsembleVoteClassifier
ens1model = EnsembleVoteClassifier(clfs=[knnmodel, nbmodel, rfmodel], weights=[1,1,1])
accuracies = cross_val_score(ens1model, X_train, Y_train, scoring='accuracy', cv=10)
ens1model.fit(X_train, Y_train)
Y_test_pred = ens1model.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Ensemble (equal, hard)', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])

print('Comparison with Ensemble (weighting):')
ens2model = EnsembleVoteClassifier(clfs=[knnmodel, nbmodel, rfmodel], weights=[1,1,2])
accuracies = cross_val_score(ens2model, X_train, Y_train, scoring='accuracy', cv=10)
ens2model.fit(X_train, Y_train)
Y_test_pred = ens2model.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Ens. (weighted, hard)', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])
    
print('Comparison with Ensemble (Soft Voting):')
ens3model = EnsembleVoteClassifier(clfs=[knnmodel, nbmodel, rfmodel], weights=[1,1,2], voting='soft')
accuracies = cross_val_score(ens3model, X_train, Y_train, scoring='accuracy', cv=10)
ens3model.fit(X_train, Y_train)
Y_test_pred = ens3model.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Ens. (weighted, soft)', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])
    
print('Ensemble with Grid Search:')
ens4model = EnsembleVoteClassifier(clfs=[knnmodel, nbmodel, rfmodel], weights=[1,1,2], voting='soft')
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'kneighborsclassifier__n_neighbors': [ 5,  7,  9],
    'randomforestclassifier__n_estimators': [ 100, 150, 200]
}
CV_ensmodel = GridSearchCV(estimator=ens4model, param_grid=param_grid, cv=10)
CV_ensmodel.fit(X_train, Y_train)
print('Best Parameters:', CV_ensmodel.best_params_)
#assign these parameters to the ensemble voting classifier
ens4model = ens4model.set_params(**CV_ensmodel.best_params_)
ens4model.fit(X_train, Y_train)
Y_test_pred = ens4model.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Ensemble (gridsearch)', 
                          CV_ensmodel.cv_results_['mean_test_score'][CV_ensmodel.best_index_], 
                          CV_ensmodel.cv_results_['std_test_score'][CV_ensmodel.best_index_], accte]
print(report.loc[len(report)-1])


####################################
# Comparing with Stacking Ensemble #
####################################

from sklearn.linear_model import LogisticRegression
lr_ensemble = LogisticRegression()

print('Comparison with Stacking based on Logistic Regression:')
from mlxtend.classifier import StackingClassifier
stens1model = StackingClassifier(classifiers=[knnmodel, nbmodel, rfmodel],
                                 use_probas=True,
                                 average_probas=False,
                                 meta_classifier=lr_ensemble)
accuracies = cross_val_score(stens1model, X_train, Y_train, scoring='accuracy', cv=10)
stens1model.fit(X_train, Y_train)
Y_test_pred = stens1model.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Stacking Ensemble', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])

print('Stacking with Grid Search:')
stens2model = StackingClassifier(classifiers=[knnmodel, nbmodel, rfmodel],
                                 use_probas=True,
                                 average_probas=False,
                                 meta_classifier=lr_ensemble)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'kneighborsclassifier__n_neighbors': [ 5,  7,  9],
    'randomforestclassifier__n_estimators': [ 100, 150, 200]
}
CV_ensmodel = GridSearchCV(estimator=stens2model, param_grid=param_grid, cv=10)
CV_ensmodel.fit(X_train, Y_train)
print('Best Parameters:', CV_ensmodel.best_params_)
stens2model = stens2model.set_params(**CV_ensmodel.best_params_)
stens2model.fit(X_train, Y_train)
Y_test_pred = stens2model.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['Stacking (gridsearch)', 
                          CV_ensmodel.cv_results_['mean_test_score'][CV_ensmodel.best_index_], 
                          CV_ensmodel.cv_results_['std_test_score'][CV_ensmodel.best_index_], accte]
print(report.loc[len(report)-1])


################
# Final Report #
################

print(report)
