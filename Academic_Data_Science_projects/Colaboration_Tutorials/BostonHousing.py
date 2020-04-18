# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import warnings filter
from warnings import simplefilter
# ignore warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

os.chdir("D:/Dropbox/Lehre/Digital Analytics/Excercises/Regression")
data_ori = pd.read_csv('BostonHousing.csv')
print(data_ori.shape)
# types
print(data_ori.dtypes)
# feature names
print(list(data_ori))
# head
print(data_ori.head(6))
# descriptions
#pd.set_option('display.expand_frame_repr', False)
#pd.set_option("display.precision", 4)
print(data_ori.describe())

# standardize data = (data_ori-data_ori.mean())/data_ori.std()
data=(data_ori-data_ori.min())/(data_ori.max()-data_ori.min())
print(data.head(6))

X = data.drop('medv', axis = 1)
Y = data['medv']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
print(Y_train.shape)
print(Y_test.shape)

Y_train_mean = Y_train.mean()
print("Y_train_mean =", Y_train_mean)
Y_train_meandev = sum((Y_train-Y_train_mean)**2)
print("Y_train_meandev =", Y_train_meandev)
Y_test_meandev = sum((Y_test-Y_train_mean)**2)
print("Y_test_meandev =", Y_test_meandev)

# create report dataframe
report = pd.DataFrame(columns=['Model','R2.Train','R2.Test'])


################
#     OLS      #
################

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_train_pred = lm.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = lm.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['OLS Regression', r2, pseudor2]


####################
# Ridge Regression #
####################

from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=2)
ridgereg.fit(X_train, Y_train)
Y_train_pred = ridgereg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = ridgereg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

# find best lambda (alphas)
r2s = np.zeros((3,21), float)
alphas = np.linspace(0, 2, 21)
for k in range(0, 21):
    ridgereg = Ridge(alpha=alphas[k])
    ridgereg.fit(X_train, Y_train)
    Y_train_pred = ridgereg.predict(X_train)
    Y_train_dev = sum((Y_train-Y_train_pred)**2)
    r2 = 1 - Y_train_dev/Y_train_meandev
    r2s[1,k] = r2
    Y_test_pred = ridgereg.predict(X_test)
    Y_test_dev = sum((Y_test-Y_test_pred)**2)
    pseudor2 = 1 - Y_test_dev/Y_test_meandev
    r2s[2,k] = pseudor2
    r2s[0,k] = alphas[k]
plt.plot(alphas, r2s[1,:])
plt.plot(alphas, r2s[2,:])
plt.xticks(alphas, rotation=90)
plt.xlabel('Alpha')
plt.ylabel('R2 / Pseudo-R2')
plt.title('Ridge Regression')
plt.show()

from tabulate import tabulate
headers = ["Alpha", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)
maxi = np.array(np.where(r2s==r2s[2:].max()))
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)

ridgereg = Ridge(alpha=r2s[0,maxi[1,:]])
ridgereg.fit(X_train, Y_train)
Y_train_pred = ridgereg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = ridgereg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Ridge Regression', r2, pseudor2]


#############################
# Support Vector Regression #
#############################

# linear kernel
from sklearn.svm import SVR
LinSVRreg = SVR(kernel='linear', C=1.0, epsilon=0.1)
LinSVRreg.fit(X_train, Y_train)
Y_train_pred = LinSVRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = LinSVRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

n = 21
r2s = np.zeros((4,n*n), float)
costs = np.linspace(0, 0.2, n)
costs[0] = 0.001
epsilons = np.linspace(0, 0.1, n)
#epsilons[0] = 0.01
row = 0
for k in range(0, n):
    for l in range(0, n):
        LinSVRreg = SVR(kernel='linear', C=costs[k], epsilon=epsilons[l])
        LinSVRreg.fit(X_train, Y_train)
        Y_train_pred = LinSVRreg.predict(X_train)
        Y_train_dev = sum((Y_train-Y_train_pred)**2)
        r2 = 1 - Y_train_dev/Y_train_meandev
        r2s[2,row] = r2
        Y_test_pred = LinSVRreg.predict(X_test)
        Y_test_dev = sum((Y_test-Y_test_pred)**2)
        pseudor2 = 1 - Y_test_dev/Y_test_meandev
        r2s[3,row] = pseudor2
        r2s[0,row] = costs[k]
        r2s[1,row] = epsilons[l]
        row = row + 1

from tabulate import tabulate
headers = ["Cost", "Epsilon", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(r2s==r2s[3:].max()))
print(maxi[1,:])
print(r2s[:,maxi[1,:]])
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[0,:]
y = r2s[1,:]
z = r2s[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Cost')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Pseudo-R2')
plt.show()

LinSVRreg = SVR(kernel='linear', C=r2s[0,maxi[1,:]], epsilon=r2s[1,maxi[1,:]])
LinSVRreg.fit(X_train, Y_train)
Y_train_pred = LinSVRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = LinSVRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Linear SVR', r2, pseudor2]

# radial kernel
RbfSVRreg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
RbfSVRreg.fit(X_train, Y_train)
Y_train_pred = RbfSVRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = RbfSVRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

n = 21
r2s = np.zeros((5,n*n*n), float)
costs = np.linspace(0, 3, n)
costs[0] = 0.001
epsilons = np.linspace(0, 0.1, n)
gammas = np.linspace(0, 4.0, n)
gammas[0] = 0.1
row = 0
for k in range(0, n):
    for l in range(0, n):
        for m in range(0, n):
            RbfSVRreg = SVR(kernel='rbf', C=costs[k], epsilon=epsilons[l], gamma=gammas[m])
            RbfSVRreg.fit(X_train, Y_train)
            Y_train_pred = RbfSVRreg.predict(X_train)
            Y_train_dev = sum((Y_train-Y_train_pred)**2)
            r2 = 1 - Y_train_dev/Y_train_meandev
            r2s[3,row] = r2
            Y_test_pred = RbfSVRreg.predict(X_test)
            Y_test_dev = sum((Y_test-Y_test_pred)**2)
            pseudor2 = 1 - Y_test_dev/Y_test_meandev
            r2s[4,row] = pseudor2
            r2s[0,row] = costs[k]
            r2s[1,row] = epsilons[l]
            r2s[2,row] = gammas[m]
            row = row + 1

from tabulate import tabulate
headers = ["Cost", "Epsilon", "Gamma", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

maxi = np.array(np.where(r2s==r2s[4:].max()))
print(maxi[1,:])
print(r2s[:,maxi[1,:]])
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[0,:]
y = r2s[2,:]
z = r2s[4,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Cost')
ax.set_ylabel('Gamma')
ax.set_zlabel('Pseudo-R2')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[1,:]
y = r2s[2,:]
z = r2s[4,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Epsilon')
ax.set_ylabel('Gamma')
ax.set_zlabel('Pseudo-R2')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[0,:]
y = r2s[1,:]
z = r2s[4,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Cost')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Pseudo-R2')
plt.show()

RbfSVRreg = SVR(kernel='rbf', C=r2s[0,maxi[1,0]], epsilon=r2s[1,maxi[1,0]], gamma=r2s[2,maxi[1,0]])
RbfSVRreg.fit(X_train, Y_train)
Y_train_pred = RbfSVRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = RbfSVRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Radial SVR', r2, pseudor2]


##################
# Neural Network #
##################

from sklearn.neural_network import MLPRegressor
NNetRreg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(10,), random_state=0)
NNetRreg.fit(X_train, Y_train)
Y_train_pred = NNetRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = NNetRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

#varying the number of hidden neurons
r2s = np.zeros((3,20), float)
for k in range(0, 20):
    NNetRreg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(k+1,), random_state=0)
    NNetRreg.fit(X_train, Y_train)
    Y_train_pred = NNetRreg.predict(X_train)
    r2 = 1 - Y_train_dev/Y_train_meandev
    r2s[1,k] = r2
    Y_test_pred = NNetRreg.predict(X_test)
    Y_test_dev = sum((Y_test-Y_test_pred)**2)
    pseudor2 = 1 - Y_test_dev/Y_test_meandev
    r2s[2,k] = pseudor2
    r2s[0,k] = k+1
plt.plot(r2s[0,:], r2s[1,:])
plt.plot(r2s[0,:], r2s[2,:])
plt.xlim(1,20)
plt.xticks(r2s[0,:])
plt.xlabel('Hidden Neurons')
plt.ylabel('Pseudo-R2')
plt.title('Neural Network')
plt.show()

from tabulate import tabulate
headers = ["Hidden Neurons", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)
maxi = np.array(np.where(r2s==r2s[2:].max()))
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)

NNetRreg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(int(r2s[0,maxi[1,:]]),), random_state=0)
NNetRreg.fit(X_train, Y_train)
Y_train_pred = NNetRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = NNetRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

#varying the number of hidden neurons and alpha
alphas = np.linspace(0, 0.001, 21)
r2s = np.zeros((4,20*21), float)
row = 0
for k in range(0, 20):
    for l in range(0, 21):
        NNetRreg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(k+1,), alpha=alphas[l], random_state=0)
        NNetRreg.fit(X_train, Y_train)
        Y_train_pred = NNetRreg.predict(X_train)
        r2 = 1 - Y_train_dev/Y_train_meandev
        r2s[2,row] = r2
        Y_test_pred = NNetRreg.predict(X_test)
        Y_test_dev = sum((Y_test-Y_test_pred)**2)
        pseudor2 = 1 - Y_test_dev/Y_test_meandev
        r2s[3,row] = pseudor2
        r2s[0,row] = k+1
        r2s[1,row] = alphas[l]
        row = row + 1

from tabulate import tabulate
headers = ["Hidden Neurons", "Alpha", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)

print(r2s[3:].max())
maxi = np.array(np.where(r2s==r2s[3:].max()))
print(maxi[0,:], maxi[1,:])
print(r2s[:,maxi[1,:]])
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[0,:]
y = r2s[1,:]
z = r2s[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Hidden Neurons')
ax.set_ylabel('Alpha')
ax.set_zlabel('Pseudo-R2')
plt.show()

NNetRreg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(int(r2s[0,maxi[1,:]]),), alpha=r2s[1,maxi[1,:]], random_state=0)
NNetRreg.fit(X_train, Y_train)
Y_train_pred = NNetRreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = NNetRreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Neural Network', r2, pseudor2]


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestRegressor
RForreg = RandomForestRegressor(n_estimators=500, random_state=0)
RForreg.fit(X_train, Y_train)
Y_train_pred = RForreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = RForreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

#varying max_depth and n_estimators
mdepth = np.linspace(4, 8, 5)
ntrees = (np.arange(20)+1)*25
r2s = np.zeros((4, 5*20), float)
row = 0
for k in range(0, 5):
    for l in range(0, 20):
        RForreg = RandomForestRegressor(max_depth=mdepth[k], n_estimators=ntrees[l], random_state=0)
        RForreg.fit(X_train, Y_train)
        Y_train_dev = sum((Y_train-Y_train_pred)**2)
        r2 = 1 - Y_train_dev/Y_train_meandev
        r2s[2,row] = r2
        Y_test_pred = RForreg.predict(X_test)
        Y_test_dev = sum((Y_test-Y_test_pred)**2)
        pseudor2 = 1 - Y_test_dev/Y_test_meandev
        r2s[3,row] = pseudor2
        r2s[0,row] = mdepth[k]
        r2s[1,row] = ntrees[l]
        row = row + 1

from tabulate import tabulate
headers = ["Max_Depth", "n_Estimators", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

print(r2s[3:].max())
maxi = np.array(np.where(r2s==r2s[3:].max()))
print(maxi[0,:], maxi[1,:])
print(r2s[:,maxi[1,:]])
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[0,:]
y = r2s[1,:]
z = r2s[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_Depth')
ax.set_ylabel('n_Estimators')
ax.set_zlabel('Pseudo-R2')
plt.show()

RForreg = RandomForestRegressor(max_depth=int(r2s[0,maxi[1,:]]), n_estimators=int(r2s[1,maxi[1,:]]), random_state=0)
RForreg.fit(X_train, Y_train)
Y_train_pred = RForreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = RForreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Random Forest', r2, pseudor2]


#####################
# Gradient Boosting #
#####################

from sklearn.ensemble import GradientBoostingRegressor
GBoostreg = GradientBoostingRegressor(random_state=0)
GBoostreg.fit(X_train, Y_train)
Y_train_pred = GBoostreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = GBoostreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

#varying max_depth and learning_rate
r2s = np.zeros((4,21*5), float)
lr = np.linspace(0, 0.4, 21)
lr[0] = 0.01
row = 0
for k in range(0, 5):
    for l in range(0, 21):
        GBoostreg = GradientBoostingRegressor(random_state=0, max_depth=k+1, learning_rate=lr[l])
        GBoostreg.fit(X_train, Y_train)
        Y_train_dev = sum((Y_train-Y_train_pred)**2)
        r2 = 1 - Y_train_dev/Y_train_meandev
        r2s[2,row] = r2
        Y_test_pred = GBoostreg.predict(X_test)
        Y_test_dev = sum((Y_test-Y_test_pred)**2)
        pseudor2 = 1 - Y_test_dev/Y_test_meandev
        r2s[3,row] = pseudor2
        r2s[0,row] = k+1
        r2s[1,row] = lr[l]
        row = row + 1

from tabulate import tabulate
headers = ["Max_Depth", "Learning_rate", "R2", "Pseudo-R2"]
table = tabulate(r2s.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

print(r2s[3:].max())
maxi = np.array(np.where(r2s==r2s[3:].max()))
print(maxi[0,:], maxi[1,:])
print(r2s[:,maxi[1,:]])
table = tabulate(r2s[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".5f")
print("\n",table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = r2s[0,:]
y = r2s[1,:]
z = r2s[3,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_Depth')
ax.set_ylabel('Learning_rate')
ax.set_zlabel('Pseudo-R2')
plt.show()

GBoostreg = GradientBoostingRegressor(max_depth=int(r2s[0,maxi[1,:]]), learning_rate=float(r2s[1,maxi[1,:]]), random_state=0)
GBoostreg.fit(X_train, Y_train)
Y_train_pred = GBoostreg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = GBoostreg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Gradient Boosting', r2, pseudor2]


##################
# Interim Report #
##################

print(report)


####################################
# Cross Validation and Grid Search #
####################################

# OLS
from sklearn.linear_model import LinearRegression
lmCV = LinearRegression()
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'fit_intercept':[True,False]
}
CV_olsmodel = GridSearchCV(estimator=lmCV, param_grid=param_grid, cv=10)
CV_olsmodel.fit(X_train, Y_train)
print(CV_olsmodel.best_params_)
lmCV = lmCV.set_params(**CV_olsmodel.best_params_)
lmCV.fit(X_train, Y_train)
Y_train_pred = lmCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = lmCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['OLS RegressionCV', r2, pseudor2]


# Ridge Regression
from sklearn.linear_model import Ridge
ridgeregCV = Ridge()
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01]
}
CV_rrmodel = GridSearchCV(estimator=ridgeregCV, param_grid=param_grid, cv=10)
CV_rrmodel.fit(X_train, Y_train)
print(CV_rrmodel.best_params_)
ridgeregCV = ridgeregCV.set_params(**CV_rrmodel.best_params_)
ridgeregCV.fit(X_train, Y_train)
Y_train_pred = ridgeregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = ridgeregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Ridge RegressionCV', r2, pseudor2]


# Support Vector Regression
from sklearn.svm import SVR
RbfSVRregCV = SVR()
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'kernel': ["linear", "rbf"], 
    'C': [1, 3, 5, 8, 10],
    'epsilon': [0.0, 0.025, 0.05, 0.075, 0.1],
    'gamma' : [0., 1., 2., 3., 4.]
}
CV_svrmodel = GridSearchCV(estimator=RbfSVRregCV, param_grid=param_grid, cv=10)
CV_svrmodel.fit(X_train, Y_train)
print(CV_svrmodel.best_params_)
RbfSVRregCV = RbfSVRregCV.set_params(**CV_svrmodel.best_params_)
RbfSVRregCV.fit(X_train, Y_train)
Y_train_pred = RbfSVRregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = RbfSVRregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Support Vector RegressionCV', r2, pseudor2]


# Neural Network
from sklearn.neural_network import MLPRegressor
NNetRregCV = MLPRegressor(solver='lbfgs', random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'hidden_layer_sizes': [(5,), (8,), (10,), (13,)],
    'alpha': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.1],
    'activation': ["logistic", "relu", "tanh"]
}
CV_nnmodel = GridSearchCV(estimator=NNetRregCV, param_grid=param_grid, cv=10)
CV_nnmodel.fit(X_train, Y_train)
print(CV_nnmodel.best_params_)
NNetRregCV = NNetRregCV.set_params(**CV_nnmodel.best_params_)
NNetRregCV.fit(X_train, Y_train)
Y_train_pred = NNetRregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = NNetRregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Neural NetworkCV', r2, pseudor2]


# Random Forest
from sklearn.ensemble import RandomForestRegressor
RForregCV = RandomForestRegressor(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 4.,  5.,  6.,  7.,  8.],
    'n_estimators': [ 10,  50,  100, 150, 200]
}
CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid=param_grid, cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)
RForregCV = RForregCV.set_params(**CV_rfmodel.best_params_)
RForregCV.fit(X_train, Y_train)
Y_train_pred = RForregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = RForregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Random ForestCV', r2, pseudor2]


# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
GBoostregCV = GradientBoostingRegressor(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 3., 4., 5.],
    'subsample': [0.7, 0.8, 0.9],
    'n_estimators': [50, 100,150],
    'learning_rate': [0.1, 0.2, 0.3]
}
CV_gbmodel = GridSearchCV(estimator=GBoostregCV, param_grid=param_grid, cv=10)
CV_gbmodel.fit(X_train, Y_train)
print(CV_gbmodel.best_params_)
GBoostregCV = GBoostregCV.set_params(**CV_gbmodel.best_params_)
GBoostregCV.fit(X_train, Y_train)
Y_train_pred = GBoostregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)
Y_test_pred = GBoostregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)
report.loc[len(report)] = ['Gradient BoostingCV', r2, pseudor2]


################
# Final Report #
################

print(report)
