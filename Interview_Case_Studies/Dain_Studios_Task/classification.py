# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from mpl_toolkits.mplot3d import Axes3D
import collections
from IPython.display import display, HTML
from termcolor import colored
import plotly.express as px
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score,classification_report,roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import lime
import lime.lime_tabular
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.utils import np_utils



class classification:

    def __init__(self, X_train,y_train,X_test,y_test,cols):
        """
        Pass the dataframe in the form of train and test seperately with column names for feature info
        """
        ## get the vectors
        self.X_train,self.y_train,self.X_test,self.y_test = X_train,y_train,X_test,y_test
        self.labels = np.unique(self.y_train).tolist()
        # reshape the values
        self.y_train = self.y_train.reshape(-1,1)
        self.y_test = self.y_test.reshape(-1,1)
        # dropping label col as it is it target variable
        self.column_name = cols
        # the labels
        self.labels = np.unique(self.y_train).tolist()
        # Making a dataframe to store model results
        self.results = pd.DataFrame(columns=['Model','Datatype','Precision','Recall','Accuracy','F1_score'])
        # Getting the cols
        self.column_name = cols
        
    def confusion_matrix(self, pred, y_test):
        
        #"For plotting matrix plot in kernel returns in the form of plot"
        cm = confusion_matrix(y_test, pred)
        print(colored("The confusion matrix is :",color = 'green', attrs=['bold']))
        print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        
        fig.colorbar(cax)
        #ax.set_xticklabels([''] + labels)
        #ax.set_yticklabels([''] + labels)
        plt.xlabel('$Predicted$')
        plt.ylabel('$True$')
        plt.show()
        
    def calc_metrics_class(self,model_name,pred,y_test,label):
        # Print's the model's performance overall
        print(colored("Generating the results wait for it....",color = 'red', attrs=['bold']))
        # Lets see the classification metrics
        precision = precision_score(pred, y_test,average='weighted')
        recall = recall_score(pred,y_test,average='weighted')
        f1 = f1_score(pred,y_test,average='weighted')
        accuracy = accuracy_score(pred,y_test)
        self.results = self.results.append({'Model':model_name,'Datatype':label,'Precision':precision,'Recall':recall,'Accuracy':accuracy ,'F1_score':f1}, ignore_index=True)
        
        # print classification report
        print(classification_report(y_test,pred))
        # Visualise the results in dataset of "test"
        print(colored("The results of your model are:",color = 'yellow', attrs=['bold']))
        print(display(HTML(self.results.to_html())))
        self.confusion_matrix(pred, y_test)
        
    def feature_importance_lime(self, model, i = 0):
        '''
        This method shows the feature importance of each set of params in getting the result
        It can be called by word index number with the model
        '''
        print()
        print("The feature importance viz for data index %d is:"%i)
        explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,feature_names= self.column_name, class_names=self.labels)
        # Predict the result of model
        predict_fn = lambda x: model.predict_proba(x).astype(float)
        # Visualise it to pictorially view
        exp = explainer.explain_instance(self.X_test[i], predict_fn, num_features=10)
        exp.show_in_notebook(show_all=False)
        
    def feature_importance_info(self, model):
        # Time to see feature importance
        print(colored("The feature importance is :",color = 'green', attrs=['bold']))
        # feature importance of the models
        feature_importance = pd.DataFrame()
        feature_importance['variable'] = self.column_name
        feature_importance['importance'] = model.feature_importances_
        # feature_importance values in descending order
        print(feature_importance.sort_values(by='importance', ascending=False).head(15))
        # By lime
        self.feature_importance_lime(model)
        
    def random_forest(self, feature_importance = False):
        print(colored("Performing modelling for Random forest",color = 'green', attrs=['bold']))
        # Create Random Forest Model
        rf_model = RandomForestClassifier(random_state=1)
        # Specifying hyperparams for the search
        param_grid = {
        'n_estimators': [75],
        'max_features': [0.1],
        'min_samples_split': [2]
        }
        grid_model = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_model.fit(self.X_train,self.y_train)
        # Fit the model with best params
        print("Best parameters =", grid_model.best_params_)
        model_clf = rf_model.set_params(**grid_model.best_params_)
        model_clf.fit(self.X_train, self.y_train)
        # Time to test the model
        # Time to test the model for test set
        print(colored("Test results for test set",color = 'yellow', attrs=['bold']))
        self.pred = model_clf.predict(self.X_test)
        self.calc_metrics_class("Random Forest",self.pred, y_test = self.y_test, label = 'test')
        # Let's see feature importance if called
        if feature_importance:
           self.feature_importance_info(model_clf)
        # Returning model
        return model_clf
    
    def logistic_regression(self):
        print(colored("Performing modelling for Logistic Regression",color = 'blue', attrs=['bold']))
        # Create logistic regression
        logistic = linear_model.LogisticRegression(max_iter = 1000)
        # Create regularization penalty space
        penalty = ['l2']
        # Create regularization hyperparameter space
        C = np.logspace(0, 4, 10)
        # Create hyperparameter options and fot it into grid search
        hyperparameters = dict(C=C, penalty=penalty)
        grid_model = GridSearchCV(estimator=logistic, param_grid=hyperparameters, cv=5, verbose=0, n_jobs=-1)
        # Fit the model and find best hyperparams
        grid_model.fit(self.X_train,self.y_train)
        print("Best parameters =", grid_model.best_params_)
        # Fit the model with best params
        model_clf = logistic.set_params(**grid_model.best_params_)
        model_clf.fit(self.X_train, self.y_train)
        # Time to test the model for test set
        print(colored("Test results for test set",color = 'yellow', attrs=['bold']))
        self.pred = model_clf.predict(self.X_test)
        self.calc_metrics_class("Logistic Regression",self.pred, y_test = self.y_test, label = 'test')
        # Returning model
        return model_clf
    
    def gradient_boost(self, feature_importance = False):

          print(colored("Performing modelling for Gradient Boosting",color = 'green', attrs=['bold']))
          # Create gradient boosting
          GradBoostClasCV = GradientBoostingClassifier(random_state=42)

          # Specifying hyperparams for the search
          model_params = {
                            "max_depth": [10],
                            "subsample": [0.9],
                            "n_estimators":[200],
                            "learning_rate": [0.01]
                          }
          # Fit the model and find best hyperparams
          grid_model = GridSearchCV(estimator=GradBoostClasCV, param_grid=model_params, cv=5, n_jobs=-1)
          grid_model.fit(self.X_train,self.y_train)

          # Fit the model with best params
          print("Best parameters =", grid_model.best_params_)
          model_clf = GradBoostClasCV.set_params(**grid_model.best_params_)
          model_clf.fit(self.X_train, self.y_train)

          # Time to test the model
          # Time to test the model for test set
          print(colored("Test results for test set",color = 'yellow', attrs=['bold']))
          self.pred = model_clf.predict(self.X_test)
          self.calc_metrics_class("Gradient Boosting",self.pred, y_test = self.y_test, label = 'test')


          # Let's see feature importance if called
          if feature_importance:
              self.feature_importance_info(model_clf)

          # Returning model
          return model_clf
    
    def XG_Boost(self, feature_importance = False):
        print(colored("Performing modelling for XG Boost Classifier",color = 'blue', attrs=['bold']))
        # Create XGB Classifier
        xg = XGBClassifier(nthread=4, seed=42)
        model_params = {
        'max_depth': [75],
        'n_estimators': [200],
        'learning_rate': [0.01]
        }
        # Fit the model and find best hyperparams
        grid_model = GridSearchCV(estimator=xg, param_grid=model_params, cv=5,scoring = 'accuracy', n_jobs=-1)
        grid_model.fit(self.X_train,self.y_train)
        # Fit the model with best params
        print("Best parameters =", grid_model.best_params_)
        model_clf = xg.set_params(**grid_model.best_params_)
        model_clf.fit(self.X_train, self.y_train)
        # Time to test the model
        # Time to test the model for test set
        print(colored("Test results for test set",color = 'yellow', attrs=['bold']))
        self.pred = model_clf.predict(self.X_test)
        self.calc_metrics_class("XG Boost",self.pred, y_test = self.y_test, label = 'test')
        # Let's see feature importance if called
        if feature_importance:
            self.feature_importance_info(model_clf)
        # Returning model
        return model_clf
    
    def Neural_Network(self, feature_importance = False):
        print(colored("Performing modelling for neural network",color = 'blue', attrs=['bold']))
        
        # we will oe hot encode categrical variables
        y_train= np_utils.to_categorical(self.y_train)
        y_test = np_utils.to_categorical(self.y_test)
        
        # Time to compile the model
        model=Sequential()
        model.add(Dense(256, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), input_dim = self.X_train.shape[1]))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(len(np.unique(y_test)), activation="softmax"))
        opt=Adam(lr=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        
        # train the model
        history = model.fit(self.X_train, y_train, epochs = 5, validation_data = (self.X_test, y_test))
        
        # plot accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        # Time to test the model for test set
        print(colored("Test results for test set",color = 'yellow', attrs=['bold']))
        pred = model.predict(self.X_test)
        
        # get the argmax to get the labels back
        pred = np.argmax(pred, axis = -1)
        self.calc_metrics_class("Neural Network", pred, y_test = self.y_test, label = 'test')
       
        # Returning model
        return model




