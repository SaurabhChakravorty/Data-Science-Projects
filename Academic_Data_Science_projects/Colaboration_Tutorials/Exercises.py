# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:35:13 2018

@author: rossb
"""


######################
# Missing Values (I) #
######################
import numpy as np
import pandas as pd
df = pd.DataFrame({'X':[34,33,65,37,89,np.nan,43,np.nan,11,np.nan,23,np.nan]})
#1
print(len(df))
#2
df.loc[df['X'].isnull()==False]
#3
len(df.loc[df['X'].isnull()])
#4
df.loc[df['X'].isnull()]=11

#######################
# Missing Values (II) #
#######################
import numpy as np
import pandas as pd
dfa = pd.DataFrame({'Name':[np.nan, "Joseph", "Martin", np.nan, "Andrea"],
                  'Sales':[15, 18, 21, 56, np.nan],
                  'Price':[34, 52, 21, np.nan, 20]})
#1
print(dfa['Sales'].mean())
#2
print(dfa.isnull().sum())
#3
print(dfa.isnull().sum(axis=1))
#4
dfb = dfa.dropna(subset=['Sales'])
#5
dfc = dfa.dropna(thresh=2) #removes all rows if the number of non-'NaN' values is less than 2


####################################
# Sampling and Pre-Processing Data #
####################################
import pandas as pd
import os
os.chdir("D:/Dropbox/Lehre/Digital Analytics/Software/Python")
data = pd.read_csv('CreditData.csv')
print(data.head(6))
print(data.shape)

# Sample
from sklearn.utils.random import sample_without_replacement
samplesize = int(len(data) * 0.1)
index = sample_without_replacement(n_population=len(data), n_samples=samplesize, 
                                   random_state=0)
data_sample = data.iloc[index,:]

# Downsample
print(data['SeriousDlqin2yrs'].value_counts())
data_majority = data[data.SeriousDlqin2yrs=='Bad']
data_minority = data[data.SeriousDlqin2yrs=='Good']
from sklearn.utils import resample
data_majority_downsampled = resample(data_majority, n_samples=len(data_minority),
                                     replace=False, random_state=0)
data_downsampled = pd.concat([data_majority_downsampled, data_minority])
print(data_downsampled['SeriousDlqin2yrs'].value_counts())

# Standardize
from sklearn import preprocessing
sscaler = preprocessing.StandardScaler()
data_standardized = data
data_standardized.iloc[:,1:12] = sscaler.fit_transform(data_standardized.iloc[:,1:12])
print(data_standardized.head(6))

# Normalize
nscaler = preprocessing.MinMaxScaler()
data_normalized = data
data_normalized.iloc[:,1:12] = nscaler.fit_transform(data_normalized.iloc[:,1:12])
print(data_normalized.head(6))

# Partition
X = data_downsampled.drop('SeriousDlqin2yrs', axis = 1)
Y = data_downsampled['SeriousDlqin2yrs']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.5, random_state=0)
print(Y_train.value_counts())
print(Y_test.value_counts())


####################
# Visualizing Data #
####################
import os
os.chdir("D:/Dropbox/Lehre/Digital Analytics/Software/Python")
data = pd.read_csv('cherry.csv')
print(data.head(6))
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(data, vars=["Girth", "Volume"])
# alt. sns.pairplot(data.iloc[:,[0,2]])
sns.distplot(data.Volume)
sns.boxplot(data=data, orient="v", palette="Set2")


########################################
# Handling Missing Values and Outliers #
########################################
import numpy as np
import pandas as pd
import os
os.chdir("D:/Dropbox/Lehre/Digital Analytics/Software/Python")
data_ori = pd.read_csv('CountryRisk.csv')
data_ori.head(10)
print(data_ori.shape)

# Search and delete all variables having more than 35 NAs
print(data_ori.isnull().sum())
data = data_ori.dropna(thresh=(len(data_ori)-35), axis=1)
print(data.isnull().sum())
print(data.shape)

# Delete all rows (countries) having NAs
data = data.dropna()
print(data.shape)

# Inspecting for outliers
import seaborn as sns
sns.boxplot(data=data, orient="h", palette="Set2")

# Setting outliers to NA
sns.boxplot(data=data['Export'], orient="v", palette="Set2")
print(data[['Country','Export']].sort_values(by='Export', ascending=0).head(10))
data.loc[191,'Export'] = np.nan
print(data.loc[191,['Country','Export']])
sns.boxplot(data=data['Import'], orient="v", palette="Set2")
print(data[['Country','Import']].sort_values(by='Import', ascending=0).head(10))
data.loc[191,'Import'] = np.nan
print(data.loc[191,['Country','Import']])
sns.boxplot(data=data['GDP'], orient="v", palette="Set2")
print(data[['Country','GDP']].sort_values(by='GDP', ascending=0).head(10))

# Delete all rows (countries) having outliers (NAs)
data = data.dropna()
print(data.shape)

# Normalize the data
print(data.head(6))
from sklearn import preprocessing
nscaler = preprocessing.MinMaxScaler()
data.iloc[:,1:16] = nscaler.fit_transform(data.iloc[:,1:16])
print(data.head(6))

