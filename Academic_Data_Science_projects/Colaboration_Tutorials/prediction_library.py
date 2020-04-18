import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import plot_model
import warnings
warnings.filterwarnings("ignore")


def create_one_hot(df, column_name):
    """One-hot encode column values. Takes the df and column name as
    input and return the df with one-hot encoded columns as output.
    """
    df[column_name] = pd.Categorical(df[column_name])
    one_hot = pd.get_dummies(df[column_name], prefix = column_name)
    # add dummies to original df:
    df = pd.concat([one_hot, df], axis = 1)
    return df

def calc_metrics(y_true, y_pred):
    """Calculates the error metrics to evaluate the model. Calculates the R2
    score, Pseudo-R2 Score, Root Mean Squared Error and Mean Absolute Error.
    """
    y_pred_meandev = np.sum((y_pred - y_true.mean()) ** 2)
    r2 = r2_score(y_true, y_pred)
    y_test_dev = np.sum((y_true - y_pred)**2)
    pseudor2 = 1 - y_test_dev/y_pred_meandev
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return r2, pseudor2, rmse, mae

def data_cleaning(df, four_split=True):
    """Clean the data according to sound principles. The steps are:
    1. One-Hot-Encode the Weather Situation and Season.
    They have four categories each.
    2. Change weekday variable 0 to 7, so we can treat weekends and weekdays
    better.
    3. Creating a custom variable called 'rushhour'. Because there is a spike
    in traffic around standard office hours (7-8 and 17-18), we want to feed
    this information to the model so that it can predict those hours better.
    4. Avoiding the dummy-variable trap by dropping one One-Hot-Encode column
    from each encoded variable.
    5. Transform cyclical data. Hour and monthly information is cyclical.
    We transform them into sines and cosines, where the angle is 360 divided
    by the number of unique values.
    6. Drop columns that do not help much to make predictions our are
    redundant, i.e., it is completely determined using another column.
    7. Convert categorial columns to the categorical data type.
    """
    # one-hot: weathersit -> creating the dummy variables
    df = create_one_hot(df, "weathersit")
    df = create_one_hot(df, "season")

    # change 0 to 7
    df['weekday'][df['weekday'] == 0] = 7

    # Custom variable: rushhour
    df["rushhour"] = df[["hr", "workingday"]].apply(lambda x: int((x[
            "workingday"] == 1 and (7 <= x["hr"] <= 8 or 17 <= x["hr"] <= 18)))
    , axis=1)

    # avoid dummy trap
    df.drop(["season_4", "weathersit_3"], axis=1, inplace = True)

    # Transform cyclical data:
    df['hr_sin'] = np.sin(df.hr*(2.*np.pi/24))
    df['hr_cos'] = np.cos(df.hr*(2.*np.pi/24))

    if four_split:
    # delete unnecessary columns; still not sure about dropping day
        df.drop(["instant", "dteday", "cnt", "hr", "weathersit", "season"],
                axis = 1, inplace=True)

    else:
        df.drop(["instant", "dteday", "registered", "hr", "casual",
                 "weathersit", "season"], axis = 1, inplace=True)

    # convert to same data type:
    cat_dtype = pd.api.types.CategoricalDtype(
             categories=[1, 2, 3, 4, 5, 6, 7], ordered=True)
    
    df[["weathersit_1", "weathersit_2", "weathersit_4", "season_1", "season_2",
        "season_3", "yr", "holiday", "workingday", "rushhour", "mnth"
        ]] = df[["weathersit_1", "weathersit_2", "weathersit_4", "yr",
        "season_1", "season_2", "season_3", "holiday", "workingday",
        "rushhour", "mnth"]].astype("category")
    
    df["weekday"].astype(cat_dtype)

    return df

def split_data_by_day(df):
    """Creates a weekday and weekend split of the dataframe. Returns them
    separately as dataframes.
    """

    # create a weekend-weekday split
    weekends = df[df['weekday'] >= 6].copy()
    weekdays = df[df['weekday'] <= 5].copy()
    #one hot weekdays
    weekdays = create_one_hot(weekdays, "weekday")
    weekends = create_one_hot(weekends, "weekday")

    weekdays[["weekday_1", "weekday_2", "weekday_3", "weekday_4",
                    "weekday_5"]] = weekdays[["weekday_1",
                    "weekday_2", "weekday_3", "weekday_4",
                    "weekday_5"]].astype("category")

    weekends[["weekday_6", "weekday_7"]] = weekends[["weekday_6",
               "weekday_7"]].astype("category")

    weekdays.drop(["weekday"], axis = 1, inplace=True)
    weekends.drop(["weekday"], axis = 1, inplace=True)

    return weekdays, weekends

def split_data_by_user(df):
    """Creates a Causal and Registered user split of the dataframe. Returns
    them separately as dataframes.
    """
    registered_users = df.drop(["casual"], axis=1)
    X_registered_users = registered_users.drop("registered", axis=1)
    y_registered_users = registered_users["registered"]
    casual_users = df.drop(["registered", "rushhour"], axis=1)
    X_casual_users = casual_users.drop("casual", axis=1)
    y_casual_users = casual_users["casual"]

    return X_registered_users, y_registered_users, X_casual_users, y_casual_users

class Regressors():
    """Create a class of regression functions.
    """

    def __init__(self, X, y):
        """Create a train-test split for our data. Also initialises variables
        to hold the mean and mean deviation.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                test_size=0.2, random_state=0)

        y_train_mean = y_train.mean()
        y_train_meandev = sum((y_train - y_train_mean) ** 2)
        y_test_meandev = sum((y_test - y_train_mean) ** 2)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_mean = y_train_mean
        self.y_train_meandev = y_train_meandev
        self.y_test_meandev = y_test_meandev

    def fitter(self, model, param_grid):
        """Fits the regression models. Returns the scores and predictions.
        """
        grid_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_model.fit(self.X_train, self.y_train)
        print("Best parameters =", grid_model.best_params_)
        model = model.set_params(**grid_model.best_params_)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 0, None)

        r2, pseudor2, rmse, mae = calc_metrics(self.y_test, y_pred)

        return [r2, pseudor2, rmse, mae, y_pred]

    def GradientBoostCV(self):
        """Function to fit a Gradient Boosted Model using GridSearchCV.
        """
        GBoostRegCV = GradientBoostingRegressor(random_state=0)
        param_grid = {
                "max_depth": [ 6., 7., 8.],
                "subsample": [0.7, 0.8, 0.9],
                "n_estimators": [1000],
                "learning_rate": [0.1]}

        return self.fitter(GBoostRegCV, param_grid)

    def MLP(self):
        """Function to fit a Multi Layer Perceptron Model using GridSearchCV.
        """
        MLPRegCV = MLPRegressor(random_state=0)
        param_grid = {'learning_rate': ["constant", "adaptive"],
                      'hidden_layer_sizes': [(30,), (50,), (70,)],
                      'alpha': [0.03, 0.01, 0.1]}

        return self.fitter(MLPRegCV, param_grid)

    def Random_Forest(self):
        """Function to fit a Random Forest Model using GridSearchCV.
        """
        RandForRegCV = RandomForestRegressor(random_state=0)
        param_grid = { 'max_depth': [ 25., 37., 50.],
                      'n_estimators': [100]}

        return self.fitter(RandForRegCV, param_grid)

    def Linear_Regression(self):
        """Function to fit a Linear Regression Model using GridSearchCV.
        """
        lmCV = LinearRegression()
        param_grid = {'fit_intercept':[True,False]}

        return self.fitter(lmCV, param_grid)

    def lstm_model(self):
        """ Defines the LSTM architecture we want to use for our model.
        - 50 Units are chosen, to make a complex-enough model.
        - Strtucture: three LSTM layers, with a Dropout layer in between
        and one Dense layer at the end.
        - Batch size small to avoid degradation in generalizability
          caused by larger batch sizes.
         - ReLu is used as the Activation function, and the optimizer is
         set to be Adam.
        """
        #initialize model
        regressor = Sequential()
        #add LSTM layers
        regressor.add(LSTM(units=50, return_sequences=True,
                           input_shape=(1, self.X_train.shape[1])))
        #apply dropout to avoid overfitting
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True,
                           input_shape=(1, self.X_train.shape[1])))
        #apply dropout to avoid overfitting
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        #add Dense layer
        regressor.add(Dense(units=1))
        #Add activation function
        regressor.add(Activation('relu'))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        return regressor


    def LSTM(self):
        """Builds a Keras Regressor model and imports the architecture.
        Fits the model and returns error metrics as output.
        """
        model = KerasRegressor(build_fn=self.lstm_model, epochs=30,
        batch_size=16, verbose=2)
        X_Train = self.X_train
        X_Test = self.X_test
        X_Train = X_Train.values.reshape((X_Train.shape[0], 1, X_Train.shape[1]))
        X_Test = X_Test.values.reshape((X_Test.shape[0], 1, X_Test.shape[1]))
        #verbose=2 will just mention the number of epochs
        model.fit(X_Train, self.y_train)
        y_pred = model.predict(X_Test)
        y_pred = np.clip(y_pred, 0, None)

        r2, pseudor2, rmse, mae = calc_metrics(self.y_test, y_pred)

        return [r2, pseudor2, rmse, mae, y_pred]
