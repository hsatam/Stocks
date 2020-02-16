from Stocks import Stocks
import numpy as np
import pandas as pd

# Set print options to ensure ndarrays are printed as readable options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# Initialize Stocks class for operations
stock = Stocks()

# Retrieve data from Quandl using Stocks class
data = stock.getQuandlInfo(Stocks.STOCK, Stocks.START_DATE, Stocks.END_DATE, Stocks.RETURNS)

# Prepare data for model (using train_test_split 95:5)
X_train, X_test, Y_train, Y_test = stock.prepareDataForModel(data)

# Build a LinearRegression Model and carryout predictions
y_pred = stock.buildLRModel(X_train, X_test, Y_train, Y_test)

data['y_pred'] = np.nan
y_pred_ctr = 0

for ctr in range((len(data.index) - len(y_pred)), len(data.index)):
	data.iat[ctr,8] = "{:.2f}".format(y_pred[y_pred_ctr])
	y_pred_ctr += 1


# Plot Closing price by Day post predictions
stock.plotClosingByDay(data, Stocks.scriptLabel)
