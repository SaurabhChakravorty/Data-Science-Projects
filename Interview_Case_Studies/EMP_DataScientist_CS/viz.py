import pandas as pd
import numpy as np
from termcolor import colored
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns


def common_columns(df1, df2):
    '''
    Gives the common columns in both the dataframe and the unique primary key information

    '''
    # Let's's see there are any common cols or not
    cols = set(df1.columns.to_list()) & set(df2.columns.to_list())
    if cols:
        print(colored("The common columns in both these tables are :", 'green', attrs=['bold']))
        print(cols)
    else: 
        print(colored("No common columns between these two tables", 'red', attrs=['bold']))
        
    flag = 0 # For checking primary key
    for i in cols:
    # if 'Col' is unique we will take it as primary key in second table
        if pd.Series(df2[i]).is_unique == True or pd.Series(df1[i]).is_unique == True:
            flag = 1
            print(colored("The unique column '{}' can be used as primary key".format(i), 'yellow', attrs=['bold']))
    if flag == 0:
            print(colored("There are no primary keys available in both the tables", 'red', attrs=['bold']))
    return cols

def replace_by_groupby(df , column_name, group_by_cols, replace_criterion):
    '''
    This function replaces n.a values based on groupy by items

    Purpose : The purpose of this is to to not add unnecessary variables or skew the distribution
    If still we dont have have values we will replace with mean , median or mode

    Input :  Input to this is column and group by columns and critrtion whether mean. median or mode

    Output is column

    '''

    ## print the n.a value initially
    print(colored('The n.a value in the columns are: {}'.format(df[column_name].isna().sum()), 'red', attrs=['bold']))

    ## category wise collection
    if replace_criterion == 'mean': 
        t = pd.DataFrame(df.groupby(group_by_cols)[column_name].agg(pd.Series.mean)).reset_index()

    elif replace_criterion == 'median':
        t = pd.DataFrame(df.groupby(group_by_cols)[column_name].agg(pd.Series.median)).reset_index()

    else:
        t = pd.DataFrame(df.groupby(group_by_cols)[column_name].agg(pd.Series.mode)).reset_index()

    ## getting the n.a values out of it
    f = df[df[column_name].isnull()][group_by_cols]
    ind = f.index

    ## merge it with previos group by item values
    f = f.merge(t, how ='left', on=group_by_cols)

    ## add it to the df
    df.loc[ind,column_name] = f[column_name]

    ## still see if there is n.a values
    print(colored('The n.a value remaining now after join and replace are: {}'.format(df[column_name].isna().sum()), 'red', attrs=['bold']))

    ## if n.a still exist impute with criterion set

    if replace_criterion == 'mean': 
        df[column_name] = df[column_name].fillna(df[column_name].mean())

    elif replace_criterion == 'median':
        df[column_name] = df[column_name].fillna(df[column_name].median())

    else:
        df[column_name] = df[column_name].fillna(df[column_name].mode()[0])


    ## finish it
    print(colored('The n.a values has now been replaced', 'green', attrs=['bold']))

    return df[column_name]



def join_tables(table1, table2, key = [], join_type = 'left', indicator = False):
    
    '''
    This function join tables with the help of merge functionality 
    
    '''
    # Let's get unique cols in these two table
    cols = set(table1.columns.to_list()) & set(table2.columns.to_list())
    
    # Get "non- unique" cols as we will do left join
    cols = [i for i in table2.columns.to_list() if i not in cols]
    
    # We need primary keys
    cols.append(key)
    
    # We will do mostly a left join
    df = pd.merge(table1, table2[cols], how=join_type, on=key, indicator=indicator)
    
    # Let's see the n.a values
    print(colored("The count of n.a values due to this join is: ", 'red', attrs=['bold']))
    print(df[cols].isnull().sum())
    
    # Shape after joining
    print(colored("The shape of table after joining is {}".format(df.shape), 'blue', attrs=['bold']))
    
    return df




class description():

    '''
    Contains all functionalities for the data description and visualization

    '''
    def __init__(self,df):
        
        '''
        Takes the dataframe and converts into features
        '''
        self.X = df
        
    def data_description(self, summary = False):
        
        '''
        Describes the whole data in nutshell
        '''
        print(colored("The number of points in this data is {} ".format(len(self.X)), 'blue', attrs=['bold']))
        print()
        print(colored("The shape of the data is {} ".format(self.X.shape), 'yellow', attrs=['bold']))
        print()
        print(colored("Let's see the data : ", 'red', attrs=['bold']))
        #print(self.X.head())
        print()
        if summary:
            print(colored("The summary of data set is : ", 'green', attrs=['bold']))
            print(self.X.describe())
            print()
        print(colored("The count of n.a values in each column is: ", 'green', attrs=['bold']))
        print(self.X.isnull().sum())
        
        
    def value_counts(self,col_name):
    
        '''
        Gives the value counts
        '''
        print(colored("The unique values in each category of {} is : ".format(col_name), 'blue', attrs=['bold']))
        print(self.X[col_name].value_counts())
        
        

        


        
        
class visualisation():
    '''
    Generates visualisation of each type of graph in one-go
    
    '''
    def __init__(self,df):
        
        '''
        Takes the dataframe and converts into features
        '''
        self.X = df
    
    def hist_plot(self,x = None, category = None):
        
        fig = px.histogram(self.X, x=x, color=category).update_xaxes(categoryorder = "total descending")
        fig.show()
        
    def heat_map(self, df = None):
        
        if df == None:
            corr = self.X.corr()
        else:
            corr = df.corr
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot = True, cbar_kws={"shrink": .5})
                                                                        
    def box_plot(self, x = None ,  y = None, color=None):
        fig = px.box(self.X, x=x, y=y, color=color)
        fig.show()

    def line_plot(self, x = None, y = None):
        fig = px.scatter(self.X, x=x, y=y)
        fig.show()
        
    def bar_plot(self, x = None ,  y = None, color=None):
        fig = px.bar(self.X, x=x, y=y, color=color)
        fig.show()
        
    def dis_plot(self, x = None , hue=None, kind='hist', fill=False):
        fig = sns.displot(self.X, x=x, hue=hue, kind=kind, fill=fill)
        
    def pair_plot(self,kind = 'reg'):
        plt.figure(figsize=(15, 5), dpi=80)
        sns.pairplot(self.X,kind='reg')
        plt.show()
        
    def density_plot(self, x = None):
        fig, ax = plt.subplots(figsize = [12, 7])
        sns.distplot(self.X[x])
        fig.show()

    
        
    