import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score, mean_squared_error
import numpy as np

def plot_grid(image_list):
    fig, axes = plt.subplots(ncols=2, nrows=2)

    for ax, img in zip(axes.flatten(), image_list):
        ax.imshow(Image.open(img))
        ax.axis('off')

    fig.set_size_inches(8, 6)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
def basic_cleaning_pre(data):
    #Some columns used in excel for pre-processing
    data.drop(['Text Conversion', 'Text Conversion.1', 'Text Conversion.2',
       'Text Conversion.3', 'Text Conversion.4', 'Text Conversion.5',
       'Text Conversion.6', 'Founded Date_Formatted', 'Text Conversion.7',
       'Text Conversion.8', 'Text Conversion.9', 'Text Conversion.10',
       'Text Conversion.11', 'Text Conversion.12', 'Text Conversion.13'], axis=1, inplace=True)
    # Drop rows marked red in Excel. Total=17
    data.drop([851, 1688, 1779, 1782, 1784, 
            1788, 1789, 1793, 1795, 1814, 
            1816, 1819, 1820, 1851, 1853, 
            1875, 1888], inplace=True)
    
    # To remove cases where there are 3 separators. Usually for Edinburgh. 
    for i in data.index:
        loc = data['Headquarters Location'][i].split(',')
        if len(loc) > 3:
            new_loc = loc[0] + ',' + loc[1] + ',' + loc[3]
            data['Headquarters Location'][i] = new_loc

    location_info = data['Headquarters Location'].str.split(', ', expand=True)
    location_info.columns = ['City', 'State', 'Country']

    data.insert(3, 'City', location_info['City'].values)
    data.insert(4, 'State', location_info['State'].values)
    data.insert(5, 'Country', location_info['Country'].values)

    data.drop('Headquarters Location', axis=1, inplace=True)
    
    
    return data


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calc_metrics(y_true, y_pred):
    """Calculates the error metrics to evaluate the model. Calculates the R2
    score, Pseudo-R2 Score, Root Mean Squared Error and Mean Absolute Error.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print("R2", r2, '\n', "RMSE", rmse, '\n', "MAE", mae, '\n', "MAPE", mape)
    
    
def plot_feat_imp(dataframe, model, cols_to_drop):
    feat_imp = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), 
                      dataframe.drop(cols_to_drop, axis=1).columns))
    feat_imp_sort = [(elem2, elem1) for elem1, elem2 in feat_imp]
    plt.figure(figsize=[10,6])
    plt.style.use('seaborn')
    plt.title('Feature Importance - Sklearn')
    plt.barh(*zip(*feat_imp_sort))
    plt.xlabel('Importance')
    plt.show()
    
def grad_boost_reg(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    calc_metrics(y_test, pred)

def grad_boost_classification(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    calc_metrics_class(pred, y_test)

def calc_metrics_class(pred, y_test):
   precision = precision_score(pred, y_test)
   recall = recall_score(pred,y_test)
   f1 = f1_score(pred,y_test)
   accuracy = accuracy_score(pred,y_test)
   print("precision", precision, '\n', "recall", recall, '\n', "f1", f1, '\n', "accuracy", accuracy)