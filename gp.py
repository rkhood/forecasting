import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared

from plots import plot_series
from sunspots import error, get_data, sklearn_formatting


def gp(train, test, t=132):
    X_train, X_test = sklearn_formatting(train, test)

    gp_kernel = 2**2 \
                + ExpSineSquared(1, 60000.0) \
                + ExpSineSquared(2, 120000.0) \
                + WhiteKernel(2.5)
    gpr = GaussianProcessRegressor(kernel=gp_kernel)

    gpr.fit(X_train, train.values)
    y_fit = gpr.predict(X_train, return_std=False)

    # predict a cycle
    y_pred = gpr.predict(X_test, return_std=False)
    rmse = error(test.values, y_pred)
    return y_fit, y_pred, rmse


if __name__ == "__main__":

    df_train, df_test = get_data()
    gauss_y, gauss_y_pred, gauss_rmse = gp(df_train, df_test)

    plot_series(df_train, df_test, gauss_y, gauss_y_pred)
