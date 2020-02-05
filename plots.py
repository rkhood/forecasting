import pandas as pd
import matplotlib.pyplot as plt


def plot_series(df_train, df_test, y, y_pred):
	fig, ax = plt.subplots(figsize=(16, 8))

	ax.plot(pd.concat([df_train, df_test]), label='Data')
	ax.plot(df_train.index, y, label='Model')
	ax.plot(df_test.index, y_pred, label='Forecast', linewidth=3)
	
	plt.legend()
	fig.show()