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
		data["y_pred_RR"].plot(label="Predicted")

		plt.legend()
		plt.show()


	def prepareDataForModel(self, data):

		#Number of days to prdict price
		forecast_time = int(30)

		# Create predicton model
		data["prediction"] = data["Close"].shift(-1)
		data.dropna(inplace=True)

		print (data.columns.values)

		# Drop columns with p-value > 0.05 based on input from OLS model to test hypothesis in RR
		data.drop(["Deliverable Quantity","High", "Spread H-L", "No. of Trades", 
			"Total Turnover", "No. of Shares", "WAP", "% Deli. Qty to Traded Qty"], axis=1, inplace=True)

		print (data.columns.values)

		self.feature_cols = list(data.columns.values.tolist())

		X = np.array(data.drop(["prediction"], 1))
		Y = np.array(data["prediction"])

		print (X)

		# Scale the data
		scale = StandardScaler()
		X = scale.fit_transform(X)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

		return X_train, X_test, Y_train, Y_test


	def printModelMetrics(self, model, X_train, Y_train, Y_test, y_pred):

		# Print beta coefficients int he same order as passed	
		print ("Coefficients : ", list(map('{:.2f}'.format,model.coef_)))

		# Print y-intercept
		print ("Intercept \t\t : \t","{:.2f}".format(model.intercept_))

		# Compute R2 score
		print ("R2 Score \t\t : \t","{:.2f}".format(model.score(X_train, Y_train)))

		# Print results of MAE
		print ("Absolute Error \t\t : \t", "{:.2f}".format(metrics.mean_absolute_error(Y_test, y_pred)))

		# Print results of MSE
		print ("MSE \t\t\t : \t", "{:.2f}".format(metrics.mean_squared_error(Y_test, y_pred)))

		# Print results of RSME
		print ("RMSE \t\t\t : \t", "{:.2f}".format(np.sqrt(metrics.mean_squared_error(Y_test, y_pred))))

		#Zip to pair feature names and coefficents together
		print ("\n\nCoefficients for attributes used in regression...")
		coeff_zip = list(zip(self.feature_cols, model.coef_))
		for ctr in range(len(coeff_zip)):
			tabStr = ''.join("\t" * int((30-len(coeff_zip[ctr][0]))/8))
			print (coeff_zip[ctr][0], tabStr, ":\t", coeff_zip[ctr][1])


	def buildRRModel(self, X_train, X_test, Y_train, Y_test):
		
		# Build a Linear Regression model
		model = LinearRegression()
		model.fit(X_train, Y_train)

		y_pred = (model.predict(X_test))

		return y_pred, model
