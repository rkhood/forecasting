import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from plots import plot_series
from sunspots import error, get_data, sklearn_formatting


class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_f=2.0):
        self.N = N
        self.width_f = width_f

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # N centres along range
        self._centres = np.linspace(X.min(), X.max(), self.N)
        self._width = self.width_f * (self._centres[1] - self._centres[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self._centres, self._width, axis=1)


def linear(train, test, t=132):
    X_train, X_test = sklearn_formatting(train, test)

    gauss_model = make_pipeline(
        GaussianFeatures(40),
        LinearRegression(),
        )
    
    gauss_model.fit(X_train, train.values)
    y_fit = gauss_model.predict(X_train)

    # predict a cycle
    y_pred = gauss_model.predict(X_test)
    rmse = error(test.values, y_pred)
    return y_fit, y_pred, rmse


if __name__ == "__main__":

    df_train, df_test = get_data()
    lin_y, lin_y_pred, lin_rmse = linear(df_train, df_test)

    plot_series(df_train, df_test, lin_y, lin_y_pred)

