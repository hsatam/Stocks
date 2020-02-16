# Import Libraries
import quandl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Stocks:

	STOCK = "BSE/BOM500570"
	START_DATE = "2019-01-01"
	END_DATE = "2019-12-31"
	RETURNS = "pandas"
	X_prediction = None


	# Setting script label to print
	scriptLabel = "TATA Motors"


	# Quandl function to retrieve data for a specified stock from a given date to a specified date as returnType (pandas)
	def getQuandlInfo(self, stock, st_date, ed_date, returnType):

		#API Key Authentication
		quandl.ApiConfig.api_key="JXZjmNDAB_YBxpxNtMCQ"

		#Data download 
		data = quandl.get(stock, start_date=st_date, end_date=ed_date, returns=returnType)

		return data

	# Plot the graph of closing by date - Adjusted Closing Price
	def plotClosingByDay(self, data, label):
		
		#Plotting data received - Original data based on what is available in Quandl
		plt.style.use("classic")
		data["Close"].plot(label=label, figsize=(16,8), title="Adjusted Closing Price")
		plt.legend()
		plt.show()


	def prepareDataForModel(self, data):

		#Number of days to prdict price
		forecast_time = int(30)

		# Create predicton model
		data["prediction"] = data["Close"].shift(-1)
		data.dropna(inplace=True)

		X = np.array(data.drop(["prediction"], 1))
		Y = np.array(data["prediction"])

		# Scale the data
		scale = StandardScaler()
		X = scale.fit_transform(X)

		# Predict for 30 days
		Stocks.X_prediction = X[-forecast_time:]

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

		return X_train, X_test, Y_train, Y_test


	def buildLRModel(self, X_train, X_test, Y_train, Y_test):
		
		# Build a Linear Regression model
		lr = LinearRegression()
		lr.fit(X_train, Y_train)


		prediction = (lr.predict(Stocks.X_prediction))

		return prediction


	# @TODO - Add the predictions to the dataframe and plot graphy
