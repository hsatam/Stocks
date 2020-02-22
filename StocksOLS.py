# Import Libraries
import quandl
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

class Stocks:

	# Initialize Linear Regression Stocks class
	def __init__(self, stock, start_date, end_date, scriptLabel):
		self.stock = stock
		self.start_date = start_date
		self.end_date = end_date
		self.returns = "pandas"
		self.X_prediction = None
		self.feature_cols = []
		self.scriptLabel = scriptLabel


	# Quandl function to retrieve data for a specified stock from a given date to a specified date as returnType (pandas)
	def getStockDetails(self):

		#API Key Authentication
		quandl.ApiConfig.api_key="JXZjmNDAB_YBxpxNtMCQ"

		#Data download 
		data = quandl.get(self.stock, start_date=self.start_date, end_date=self.end_date, returns=self.returns)

		return data

	# Plot the graph of closing by date - Adjusted Closing Price
	def plotClosingByDay(self, data, model):

		#Plotting data received - Original data based on what is available in Quandl
		plt.style.use("classic")
		data["Close"].plot(label=self.scriptLabel, figsize=(16,8), title="Adjusted Closing Price")
		data["y_pred_OLS"].plot(label="Predicted")

		plt.legend()
		plt.show()


	def prepareDataForModel(self, data):

		#Number of days to prdict price
		forecast_time = int(30)

		# Create predicton model
		data["prediction"] = data["Close"].shift(-1)
		data.dropna(inplace=True)

		# Drop columns with p-value > 0.05
		data.drop(["Deliverable Quantity","High", "Spread H-L", "No. of Trades", 
			"Total Turnover", "No. of Shares", "WAP", "% Deli. Qty to Traded Qty"], axis=1, inplace=True)

		print (data.columns)

		Stocks.feature_cols = list(data.columns.values.tolist())

		X = np.array(data.drop(["prediction"], 1))
		Y = np.array(data["prediction"])

		# Scale the data
		scale = StandardScaler()
		X = scale.fit_transform(X)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

		return X_train, X_test, Y_train, Y_test


	def printModelMetrics(self, model, X_train, Y_train, Y_test, y_pred):

		print(model.summary())


	def buildOLSModel(self, X_train, X_test, Y_train, Y_test):

		# Build a Regression model using OLS (Ordinary Least Squares)
		X_train = sm.add_constant(X_train)      ## let's add an intercept (beta_0) to our model
		X_test  = sm.add_constant(X_test)
		model = sm.OLS(Y_train, X_train).fit()  ## sm.OLS(output, input)

		y_pred = model.predict(X_test)

		return y_pred, model

