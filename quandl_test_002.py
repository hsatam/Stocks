from Stocks import Stocks
import numpy as np

# Set print options to ensure ndarrays are printed as readable options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# Initialize Stocks class for operations
stock = Stocks()

# Retrieve data from Quandl using Stocks class
data = stock.getQuandlInfo(Stocks.STOCK, Stocks.START_DATE, Stocks.END_DATE, Stocks.RETURNS)

# Plot Closing price by Day prior to predictions
#stock.plotClosingByDay(data, Stocks.scriptLabel)

#print (Stocks.X_prediction, "\n\n")

# Prepare data for model (using train_test_split 95:5)
X_train, X_test, Y_train, Y_test = stock.prepareDataForModel(data)

#print (Stocks.X_prediction, "\n\n")

# Build a LinearRegression Model and carryout predictions
prediction = stock.buildLRModel(X_train, X_test, Y_train, Y_test)

# Plot Closing price by Day post predictions
#stock.plotClosingByDay(data, Stocks.scriptLabel)
