
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as std
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import collections
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from xgboost.sklearn import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error
from sklearn import metrics
from termcolor import colored
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, HTML
import statsmodels.formula.api as smf
import statsmodels.api as sm



class regression():
    """
    This is done here for doing regression and showing the output of regression with the help of parameters
    """

    def __init__(self, train, test, cols, split_ratio=0.3):
       '''
       Inititialise the parameters with class parameters

       '''

       # Making into a class variable and giving proper shape by splitting
       self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(train, test, test_size=split_ratio, shuffle=True)
       self.y_train = self.y_train.flatten()
       self.y_test  = self.y_test.flatten()

       # Making a dataframe to store model results
       self.results = pd.DataFrame(columns=['Model','MAE','MSE','RMSE','R_Squared'])

       # Getting the cols
       self.column_name = cols
       print(colored('The len of train data is {}'.format(len(self.X_train)),color = 'yellow', attrs=['bold']))
       print(colored('The len of test data is {}'.format(len(self.X_test)),color = 'yellow', attrs=['bold']))

    
    def result_plots(self,model,error_plots = False):

        '''
        This method shows the residual and error distribution plots of the regression parameters
        along with results of each model
        '''
        # Visualise the results in dataset of "test"
        print(colored("The results of your model are:",color = 'yellow', attrs=['bold']))
        print(display(HTML(self.results.to_html())))

        if error_plots:
            #self.y_train = self.y_train.flatten()
            print(colored("The residual and error plots",color = 'red', attrs=['bold']))
            # For visualizing residual plot
            visualizer = residuals_plot(model,self.X_train, self.y_train)

            # For visualizing error plot
            visualizer = prediction_error(model,self.X_train, self.y_train)
    
    def metric_calc(self, model_name):

        # Calculate metrics and append it
        print()
        print(colored("The metrics of regression are :",color = 'green', attrs=['bold']))
        mae  = metrics.mean_absolute_error(self.y_test, self.pred)
        mse  = metrics.mean_squared_error(self.y_test, self.pred)
        rmse = np.sqrt(metrics.mean_squared_error(self.y_test, self.pred))
        r2   = metrics.r2_score(self.y_test, self.pred)
        
        # print the metrics
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:',  mse)
        print('Root Mean Squared Error:', rmse)
        print('R Squared:', r2)

        self.results = self.results.append({'Model':model_name,'MAE':mae,'MSE':mse,'RMSE':rmse,'R_Squared':r2}, ignore_index=True)
        


    def feature_importance_info(self, model):
      # Time to see feature importance
      print()
      print(colored("The feature importance are :",color = 'yellow', attrs=['bold']))

      # feature importance of the models
      feature_importance = pd.DataFrame()
      feature_importance['variable']   = self.column_name
      feature_importance['importance'] = model.feature_importances_

      # feature_importance values in descending order
      print(feature_importance.sort_values(by='importance', ascending=False).head(15))

    def linear_regression(self):

        # Let us go with a linear regression model for simple analysis
        model = sm.OLS(self.y_train, self.X_train).fit()

        # Generate model summary
        print(model.summary())

        # Let us predict and see the accuracy
        self.pred = model.predict(self.X_test)
        self.metric_calc("Linear Regression")

        # Returning model
        return model


    def random_forest(self, feature_importance = False):
   
      print(colored("Performing modelling for Random forest",color = 'yellow', attrs=['bold']))
      # Create Random Forest Model
      rf_model = RandomForestRegressor()
      # Specifying hyperparams for the search
      param_grid = {
                    'n_estimators': [10,20, 25],
                    'max_depth':    [15, 10 ,15],
                    'min_samples_split': [10, 15],
                    'min_samples_leaf' : [2,5],
                    'bootstrap': [True, False]
                    }
      # Fit the model and find best hyperparams
      grid_model = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
      grid_model.fit(self.X_train,self.y_train)

      # Fit the model with best params
      print()
      print("Best parameters =", grid_model.best_params_)
      model_clf = rf_model.set_params(**grid_model.best_params_)
      model_clf.fit(self.X_train, self.y_train)
    
      ## reviewing the results
      cv_results = pd.DataFrame(grid_model.cv_results_)
      #print(cv_results)
        
      # Time to test the model
      self.pred = model_clf.predict(self.X_test)
      self.metric_calc("Random Forest")

      # Let's see feature importance if called
      if feature_importance:
          self.feature_importance_info(model_clf)
      

      # Returning model
      return model_clf

    def XG_Boost(self, feature_importance = False):
   
      print(colored("Performing modelling for XG Boost Regressor",color = 'yellow', attrs=['bold']))
      # Create XG Boost Model
      rf_model = XGBRegressor()
      # Specifying hyperparams for the search
      param_grid =  {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }
      # Fit the model and find best hyperparams
      grid_model = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
      grid_model.fit(self.X_train,self.y_train)
        
      ## reviewing the results
      cv_results = pd.DataFrame(grid_model.cv_results_)
      print(cv_results)
    
      # Fit the model with best params
      print()
      print("Best parameters =", grid_model.best_params_)
      model_clf = rf_model.set_params(**grid_model.best_params_)
      model_clf.fit(self.X_train, self.y_train)

      # Time to test the model
      self.pred = model_clf.predict(self.X_test)
      self.metric_calc("XG Boost")

      # Let's see feature importance if called
      if feature_importance:
          self.feature_importance_info(model_clf)
      

      # Returning model
      return model_clf
 
    def SVR_regression(self):
   
      print(colored("Performing modelling for Support Vector Regression",color = 'yellow', attrs=['bold']))
      # Create SVR model
      rf_model = SVR()
      # Specifying hyperparams for the search
      param_grid = {
                    'C'     : [0.01,0.1,1,10],
                    'gamma' : [0.01,0.1,1],
                    'kernel': ['rbf','poly','sigmoid']
                    }
      # Fit the model and find best hyperparams
      grid_model = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
      grid_model.fit(self.X_train,self.y_train)

      # Fit the model with best params
      print()
      print("Best parameters =", grid_model.best_params_)
      model_clf = rf_model.set_params(**grid_model.best_params_)
      model_clf.fit(self.X_train, self.y_train)

      # Time to test the model
      self.pred = model_clf.predict(self.X_test)
      self.metric_calc("Support Vector Machine")

      # Returning model
      return rf_model
