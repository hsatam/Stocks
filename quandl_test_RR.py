from StocksRR import Stocks
import numpy as np
import pandas as pd

# Set print options to ensure ndarrays are printed as readable options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# Retrieve data from Quandl using Stocks class
stock = "BSE/BOM500570"
start_date = "2010-01-01"
end_date = "2019-12-31"
scriptLabel = "TATA Motors"

# Initialize Stocks class for operations
stock = Stocks(stock, start_date, end_date, scriptLabel)
model = ['RR']

data = stock.getStockDetails()

# Prepare data for model (using train_test_split 95:5)
X_train, X_test, Y_train, Y_test = stock.prepareDataForModel(data)

for mdl in model:
	pred_col = 'y_pred_' + mdl
	data[pred_col] = np.nan

	# Build a LinearRegression Model and carryout predictions
	pred_col, model = stock.buildRRModel(X_train, X_test, Y_train, Y_test)

	y_pred_ctr = 0

	for ctr in range((len(data.index) - len(pred_col)), len(data.index)):
		data.iat[ctr,(len(data.columns) - 1)] = "{:.2f}".format(pred_col[y_pred_ctr])
		y_pred_ctr += 1

	stock.printModelMetrics (model, X_train, Y_train, Y_test, pred_col)

	# Plot Closing price by Day post predictions
	stock.plotClosingByDay(data, model)
