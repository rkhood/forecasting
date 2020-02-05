import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from sklearn.preprocessing import MinMaxScaler

from plots import plot_series
from sunspots import error, get_data

np.random.seed(7)


def lag_data(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    return pd.concat(columns, axis=1)


def difference(data, interval=1):
    return pd.Series(np.diff(data, interval))


def invert_difference(history, yhat, interval=1):
    return np.hstack((history[0], yhat)).cumsum()
 

def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    
    def trans(data):
        data = data.reshape(data.shape[0], data.shape[1])
        return scaler.transform(data)
    return scaler, trans(train), trans(test)


def invert_scale(scaler, yhat): 
    return scaler.inverse_transform(yhat)[:, -1]


def convert_data(df, ts=132):
	diffed = difference(df.values.reshape(-1), 1)
	lagged = lag_data(diffed, ts)
	sup = lagged.values[ts:, :]

	train, test = sup[:-ts, :], sup[-ts:, :]
	scaler, train_scaled, test_scaled = scale(train, test)

	X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

	X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]
	X_test = X_test.reshape(X_test.shape[0], ts, 1)
	return X_train, X_test, y_train, y_test


def invert_data(data, X, scaler, ts=132, batch_size=1):
	y = model.predict(X, batch_size=batch_size)
	y = invert_scale(scaler, np.hstack([X.reshape(-1, ts), y]))
	y = invert_difference(data, y)
	return y


def lstm_fn(df, ts=132, batch_size=1):
	df = convert_data(df)

	model = Sequential()
	model.add(
		LSTM(16,
			batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
			stateful=True,
		))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')

	for i in range(100):
	    print(i)
	    model.fit(
	    	X_train,
	    	y_train,
	    	epochs=1,
	    	batch_size=batch_size,
	    	verbose=0,
	    	shuffle=False,
	    	)
	    model.reset_states()

	y_fit = invert_data(df.values[ts:-ts], X_train, scaler)
	y_pred = invert_data(df.values[-ts:], X_test, scaler)
	rmse = error(df.values[-ts:], y_pred[1:])
	return y_fit, y_pred[1:], rmse


if __name__ == '__main__':

    df_train, df_test = get_data()
    lstm_y, lstm_y_pred, lstm_rmse = lstm_fn(pd.concat([df_train, df_test]))

    plot_series(df_train, df_test, lstm_y, lstm_y_pred)
