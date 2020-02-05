import pandas as pd

from fbprophet import Prophet

from sunspots import error, get_data


def prophet(train, test, t=132):
    df = pd.DataFrame(data={"ds": train.index, "y": train.values.reshape(-1)})

    m = Prophet()
    m.add_seasonality(name='11yr', period=365.25 * 11, fourier_order=5)
    m.fit(df)

    # predict a cycle
    future = m.make_future_dataframe(periods=t)
    forecast = m.predict(future)
    y_pred = forecast['yhat'][-t:]
    rmse = error(test.values, y_pred)
    return [m, forecast], y_pred, rmse


if __name__ == "__main__":

    df_train, df_test = get_data()
    prophet_y, prophet_y_pred, prophet_rmse = prophet(df_train, df_test)

    f1 = prophet_y[0].plot(prophet_y[1])
    f1.show()
    f2 = prophet_y[0].plot_components(prophet_y[1])
    f2.show()
