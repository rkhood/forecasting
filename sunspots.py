import pandas as pd
import numpy as np

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def read_data(f='SN_d_tot_V2.0.csv'):
    df = pd.read_csv(
        f,
        sep=';',
        header=None,
        na_values='-1',
        usecols=[0, 1, 2, 4],
        parse_dates=[[0, 1, 2]],
        )
    df.columns = ['date', 'sunspots']
    df.set_index('date', inplace=True)
    return df


def clean_data(df):
    df[df.sunspots == -1] = np.nan
    df = df.loc['1850':]
    df = df.resample('1m').mean()
    df = (df - df.mean()) / df.std()
    
    df.sunspots = savgol_filter(
        df.sunspots,
        window_length=45,
        polyorder=10,
        )
    return df


def get_data(t='2009-04-30'):
    df = read_data()
    df = clean_data(df)
    # train, test
    return df.loc[:t], df.loc[t:].iloc[1:]


def error(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: %.3f' % (rmse))
    return rmse


def sklearn_formatting(train, test):
    def format_fn(data):
        return (data.index - data.index[0]).days.values.reshape(-1, 1)

    X_train = format_fn(train)
    X_test = format_fn(test) + X_train[-1]
    return X_train, X_test
