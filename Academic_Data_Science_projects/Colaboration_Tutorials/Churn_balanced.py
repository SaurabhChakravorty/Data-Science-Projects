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


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

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

# normalize
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
print(X.head(6))

# Partition into Training and Test sampe
evens = [n for n in range(X_ori.shape[0]) if n % 2 == 0]
X_train = X.iloc[evens,:]
Y_train = Y.iloc[evens]
print(Y_train.value_counts())
odds = [n for n in range(X_ori.shape[0]) if n % 2 != 0]
X_test = X.iloc[odds,:]
Y_test = Y.iloc[odds]
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
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = knnmodel.predict(X_test)
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
print('Optimal k =', opt_k)

knnmodel = KNeighborsClassifier(n_neighbors=opt_k)
knnmodel.fit(X_train, Y_train)
Y_train_pred = knnmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
Y_test_pred = knnmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['k-NN', acctr, accte]
print(report)

#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

Y_train_pred = knnmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
np.set_printoptions(precision=2)
class_names = ['no', 'yes']
plt.figure()
plot_confusion_matrix(cmtr, classes=class_names, title='KNN train')

#calculate f1 score
from sklearn.preprocessing import LabelEncoder
lb_churn = LabelEncoder()
Y_test_code = lb_churn.fit_transform(Y_test)
Y_test_pred_code = lb_churn.fit_transform(Y_test_pred)
from sklearn.metrics import f1_score
f1te = f1_score(Y_test_code, Y_test_pred_code)
print(f1te)

#calculate ROC and AUC and plot the curve
Y_probs = knnmodel.predict_proba(X_test)
print(Y_probs[0:6,:])
Y_test_probs = np.array(np.where(Y_test=='yes', 1, 0))
print(Y_test_probs[0:6])
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(Y_test_probs, Y_probs[:, 1])
print (fpr, tpr, threshold)
from sklearn.metrics import auc
roc_auc = auc(fpr, tpr)
print(roc_auc)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of k-NN')
plt.show()


###############
# Naive Bayes #
###############

from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
nbmodel.fit(X_train, Y_train)

Y_train_pred = nbmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = nbmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Naive Bayes', acctr, accte]


######################
#   Decision Trees   #
######################

from sklearn.tree import DecisionTreeClassifier
etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0)
etmodel.fit(X_train, Y_train)
Y_train_pred = etmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = etmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#predict probabilities instead of class assignments
Y_train_pred_prob = etmodel.predict_proba(X_train)

#find optimal max_depth
accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=k+1)
    etmodel.fit(X_train, Y_train)
    Y_train_pred = etmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = etmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Entropy)')
plt.show()

etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=5)
etmodel.fit(X_train, Y_train)
Y_train_pred = etmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = etmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Tree (Entropy)', acctr, accte]

#show tree using graphviz
import graphviz 
dot_data = sk.tree.export_graphviz(etmodel, out_file=None,
                         feature_names=list(X),  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.format = 'png'
graph.render("Churn_entropy") 


#     Gini      #
from sklearn.tree import DecisionTreeClassifier
gtmodel = DecisionTreeClassifier(random_state=0)
gtmodel.fit(X_train, Y_train)
Y_train_pred = gtmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gtmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    gtmodel = DecisionTreeClassifier(random_state=0, max_depth=k+1)
    gtmodel.fit(X_train, Y_train)
    Y_train_pred = gtmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = gtmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Gini)')
plt.show()

gtmodel = DecisionTreeClassifier(random_state=0, max_depth=8)
gtmodel.fit(X_train, Y_train)
Y_train_pred = gtmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gtmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Tree (Gini)', acctr, accte]

import graphviz 
dot_data = sk.tree.export_graphviz(gtmodel, out_file=None,
                         feature_names=list(X),  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.format = 'png'
graph.render("Churn_gini") 

#show feature importance
list(zip(X, gtmodel.feature_importances_))
index = np.arange(len(gtmodel.feature_importances_))
bar_width = 1.0
plt.bar(index, gtmodel.feature_importances_, bar_width)
plt.xticks(index,  list(X), rotation=90) # labels get centered
plt.show()


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = rfmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#varying max_depth
accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    rfmodel = RandomForestClassifier(random_state=0, max_depth=k+1)
    rfmodel.fit(X_train, Y_train)
    Y_train_pred = rfmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = rfmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

#varying n_estimators
accuracies = np.zeros((2,20), float)
ntrees = (np.arange(20)+1)*10
for k in range(0, 20):
    rfmodel = RandomForestClassifier(random_state=0, n_estimators=ntrees[k])
    rfmodel.fit(X_train, Y_train)
    Y_train_pred = rfmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = rfmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(ntrees, accuracies[0,:])
plt.plot(ntrees, accuracies[1,:])
plt.xticks(ntrees, rotation=90)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

#varying max_depth and n_estimators
mdepth = np.linspace(4, 8, 5)
accuracies = np.zeros((4,5*20), float)
row = 0
for k in range(0, 5):
    for l in range(0, 20):
        rfmodel = RandomForestClassifier(random_state=0, max_depth=mdepth[k], n_estimators=ntrees[l])
        rfmodel.fit(X_train, Y_train)
        Y_train_pred = rfmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = rfmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = mdepth[k]
        accuracies[1,row] = ntrees[l]
        row = row + 1

from tabulate import tabulate
headers = ["Max_Depth", "n_Estimators", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

print(accuracies[3].max())
maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[0,:], maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0,:]
y = accuracies[1,:]
z = accuracies[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_Depth')
ax.set_ylabel('n_Estimators')
ax.set_zlabel('accte')
plt.show()

rfmodel = RandomForestClassifier(random_state=0, max_depth=7, n_estimators=60)
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = rfmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Random Forest', acctr, accte]

# View a list of the features and their importance scores
list(zip(X_train, rfmodel.feature_importances_))


################################
# Gradient Boosting Classifier #
################################

from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0)
gbmodel.fit(X_train, Y_train)
Y_train_pred = gbmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gbmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

#varying max_depth
accuracies = np.zeros((2,10), float)
for k in range(0, 10):
    gbmodel = GradientBoostingClassifier(random_state=0, max_depth=k+1)
    gbmodel.fit(X_train, Y_train)
    Y_train_pred = gbmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0,k] = acctr
    Y_test_pred = gbmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 11), accuracies[0,:])
plt.plot(range(1, 11), accuracies[1,:])
plt.xlim(1,10)
plt.xticks(range(1, 11))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting')
plt.show()

#varying learning_rate
accuracies = np.zeros((3,21), float)
lr = np.linspace(0, 0.4, 21)
lr[0] = 0.01
for k in range(0, 21):
    gbmodel = GradientBoostingClassifier(random_state=0, learning_rate=lr[k])
    gbmodel.fit(X_train, Y_train)
    Y_train_pred = gbmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = gbmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = lr[k]
plt.plot(lr, accuracies[1,:])
plt.plot(lr, accuracies[2,:])
plt.xlabel('Learning_rate')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting')
plt.show()

#varying max_depth and learning_rate
accuracies = np.zeros((4,21*10), float)
lr = np.linspace(0, 0.4, 21)
lr[0] = 0.01
row = 0
for k in range(0, 10):
    for l in range(0, 21):
        gbmodel = GradientBoostingClassifier(random_state=0, max_depth=k+1, learning_rate=lr[l])
        gbmodel.fit(X_train, Y_train)
        Y_train_pred = gbmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = gbmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = k+1
        accuracies[1,row] = lr[l]
        row = row + 1

from tabulate import tabulate
headers = ["Max_depth", "Learning_rate", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0,:]
y = accuracies[1,:]
z = accuracies[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_depth')
ax.set_ylabel('Learning_rate')
ax.set_zlabel('accte')
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0, max_depth=4, learning_rate=0.2)
gbmodel.fit(X_train, Y_train)
Y_train_pred = gbmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = gbmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Gradient Boosting', acctr, accte]


#########################
# Discriminant Analysis #
#########################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
dismodel = LinearDiscriminantAnalysis()
dismodel.fit(X_train, Y_train)
Y_train_pred = dismodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = dismodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Linear Discriminant Analysis', acctr, accte]

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qdismodel = QuadraticDiscriminantAnalysis()
qdismodel.fit(X_train, Y_train)
Y_train_pred = qdismodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = qdismodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Quadratic Discriminant Analysis', acctr, accte]


#######################
# Logistic Regression #
#######################

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
lrmodel.fit(X_train, Y_train)

Y_train_pred = lrmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = lrmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Logistic Regression', acctr, accte]


##################
# Neural Network #
##################

from sklearn.neural_network import MLPClassifier
nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(17,), random_state=0)
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

accuracies = np.zeros((3,20), float)
for k in range(0, 20):
    nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(k+1,), random_state=0)
    nnetmodel.fit(X_train, Y_train)
    Y_train_pred = nnetmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = nnetmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = k+1
plt.plot(range(1, 21), accuracies[1,:])
plt.plot(range(1, 21), accuracies[2,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Hidden Neurons')
plt.ylabel('Accuracy')
plt.title('Neural Network')
plt.show()

from tabulate import tabulate
headers = ["Hidden Neurons", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)
maxi = np.array(np.where(accuracies==accuracies[2:].max()))
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7,), random_state=0)
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Neural Network', acctr, accte]


##########################
# Support Vector Machine #
###########################

#linear kernel
from sklearn.svm import SVC
LinSVCmodel = SVC(kernel='linear', C=10, random_state=0)
LinSVCmodel.fit(X_train, Y_train)
Y_train_pred = LinSVCmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = LinSVCmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['SVM (Linear)', acctr, accte]

accuracies = np.zeros((3,21), float)
costs = np.linspace(0, 40, 21)
costs[0] = 0.5
for k in range(0, 21):
    LinSVCmodel = SVC(kernel='linear', C=costs[k], random_state=0)
    LinSVCmodel.fit(X_train, Y_train)
    Y_train_pred = LinSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = LinSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = costs[k]
plt.plot(costs, accuracies[1,:])
plt.plot(costs, accuracies[2,:])
plt.xlim(1,20)
plt.xticks(costs, rotation=90)
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Linear SVM')
plt.show()

from tabulate import tabulate
headers = ["Cost", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#radial kernel
from sklearn.svm import SVC
accuracies = np.zeros((3,21), float)
costs = np.linspace(0, 40, 21)
costs[0] = 0.5
for k in range(0, 21):
    RbfSVCmodel = SVC(kernel='rbf', C=costs[k], gamma=0.2, random_state=0)
    RbfSVCmodel.fit(X_train, Y_train)
    Y_train_pred = RbfSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = RbfSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = costs[k]
plt.plot(costs, accuracies[1,:])
plt.plot(costs, accuracies[2,:])
plt.xlim(1,20)
plt.xticks(costs, rotation=90)
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Radial SVM')
plt.show()

from tabulate import tabulate
headers = ["Cost", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

accuracies = np.zeros((3,21), float)
gammas = np.linspace(0, 4.0, 21)
gammas[0] = 0.1
for k in range(0, 21):
    RbfSVCmodel = SVC(kernel='rbf', C=1, gamma=gammas[k], random_state=0)
    RbfSVCmodel.fit(X_train, Y_train)
    Y_train_pred = RbfSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1,k] = acctr
    Y_test_pred = RbfSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2,k] = accte
    accuracies[0,k] = gammas[k]
plt.plot(gammas, accuracies[1,:])
plt.plot(gammas, accuracies[2,:])
plt.xticks(gammas, rotation=90)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Radial SVM')
plt.show()

from tabulate import tabulate
headers = ["Gamma", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

n = 21
accuracies = np.zeros((4,n*n), float)
costs = np.linspace(0, 20, n)
costs[0] = 0.5
gammas = np.linspace(0, 4.0, n)
gammas[0] = 0.1
row = 0
for k in range(0, n):
    for l in range(0, n):
        RbfSVCmodel = SVC(kernel='rbf', C=costs[k], gamma=gammas[l], random_state=0)
        RbfSVCmodel.fit(X_train, Y_train)
        Y_train_pred = RbfSVCmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = RbfSVCmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = costs[k]
        accuracies[1,row] = gammas[l]
        row = row + 1

from tabulate import tabulate
headers = ["Cost", "Gamma", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0,:]
y = accuracies[1,:]
z = accuracies[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Cost')
ax.set_ylabel('Gamma')
ax.set_zlabel('accte')
plt.show()

RbfSVCmodel = SVC(kernel='rbf', C=14, gamma=0.4 , random_state=0)
RbfSVCmodel.fit(X_train, Y_train)
Y_train_pred = RbfSVCmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = RbfSVCmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['SVM (Radial)', acctr, accte]


#polynomial kernel
n = 21
accuracies = np.zeros((4,n*n), float)
costs = np.linspace(0, 20, n)
costs[0] = 0.5
degrees = np.linspace(0, 10.0, n)
degrees[0] = 0.1
row = 0
for k in range(0, n):
    for l in range(0, n):
        PolySVCmodel = SVC(kernel='poly', C=costs[k], degree=degrees[l], random_state=0)
        PolySVCmodel.fit(X_train, Y_train)
        Y_train_pred = PolySVCmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2,row] = acctr
        Y_test_pred = PolySVCmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = costs[k]
        accuracies[1,row] = gammas[l]
        row = row + 1

from tabulate import tabulate
headers = ["Cost", "Degree", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(accuracies==accuracies[3:].max()))
print(maxi[1,:])
print(accuracies[:,maxi[1,:]])
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)


################
# Final Report #
################

print(report)
