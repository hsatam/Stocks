from Stocks import Stocks
import numpy as np
import pandas as pd

# Set print options to ensure ndarrays are printed as readable options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# Initialize Stocks class for operations
stock = Stocks()
model = ['LR']

# Retrieve data from Quandl using Stocks class
data = stock.getStockDetails(Stocks.STOCK, Stocks.START_DATE, Stocks.END_DATE, Stocks.RETURNS)

# Prepare data for model (using train_test_split 95:5)
X_train, X_test, Y_train, Y_test = stock.prepareDataForModel(data)

for mdl in model:
	pred_col = 'y_pred_' + mdl
	data[pred_col] = np.nan

	if mdl == 'LR':
		# Build a LinearRegression Model and carryout predictions
		pred_col, model = stock.buildLRModel(X_train, X_test, Y_train, Y_test)
	elif mdl == "OLS":
		# Build a OLS Regression Model and carryout predictions
		pred_col, model = stock.buildOLSModel(X_train, X_test, Y_train, Y_test)


	y_pred_ctr = 0

	for ctr in range((len(data.index) - len(pred_col)), len(data.index)):
		data.iat[ctr,8] = "{:.2f}".format(pred_col[y_pred_ctr])
		y_pred_ctr += 1


	stock.printModelMetrics (model, X_train, Y_train, Y_test, pred_col)

	# Plot Closing price by Day post predictions
	stock.plotClosingByDay(data, Stocks.scriptLabel, mdl)
