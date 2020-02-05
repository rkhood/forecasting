import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

from plots import plot_series
from sunspots import error, get_data


def stationary(df):
	return (df - df.shift()).dropna()


def autocorr(data):
	fig, ax = plt.subplots(2, figsize=(15,8))
	plot_acf(data.values, ax=ax[0], lags=30)
	plot_pacf(data.values, ax=ax[1], lags=30)


def fit_arima_params(data):
	# fit ARIMA(p, d, q) model using aic
	best_aic = np.inf 
	best_order = None
	best_mdl = None

	pq_rng = range(12)
	d_rng = range(5)
	for i in pq_rng:
	    for d in d_rng:
	        for j in pq_rng:
	            try:
	                tmp_mdl = ARIMA(data, order=(i,d,j)).fit()
	                tmp_aic = tmp_mdl.aic
	                if tmp_aic < best_aic:
	                    best_aic = tmp_aic
	                    best_order = (i, d, j)
	                    best_mdl = tmp_mdl
	            except: continue
	print('aic: %6.2f -- order: %s'%(best_aic, best_order))
	return best_aic, best_order


def arima_fn(train, test, order=(10, 1, 9)):
	# fit ARIMA with calculated order from above
	model = ARIMA(train.values, order=order)
	res = model.fit(disp=-1)

	y_fit = pd.Series(
	res.fittedvalues,
	copy=True,
	index=train.index[1:],
	).cumsum()

	# predict a cycle
	y_pred = res.predict(
		start=len(train),
		end=len(train) + len(test) - 1,
		).cumsum()
	rmse = error(test.values, y_pred)
	return y_fit, y_pred, rmse


if __name__ == "__main__":

	df_train, df_test = get_data()
	df_train = stationary(df_train)
	df_test = stationary(df_test)

	arima_y, arima_y_pred, arima_rmse = arima_fn(df_train, df_test)

	plot_series(df_train.iloc[1:], df_test, arima_y, arima_y_pred)
