# Import Libraries
import quandl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def getQuandlInfo(stock, st_date, ed_date, returnType):
	#Data download 
	data = quandl.get(stock, start_date=st_date, end_date=ed_date, returns=returnType)

	return data


#API Key Authentication
quandl.ApiConfig.api_key="JXZjmNDAB_YBxpxNtMCQ"

#Data download 
STOCK = "BSE/BOM500570"
START_DATE = "2019-01-01"
END_DATE = "2019-12-31"
RETURNS = "pandas"

#Number of days to prdict price
forecast_time = int(30)

data = getQuandlInfo(STOCK, START_DATE, END_DATE, RETURNS)

#print info of data returned
print (data.tail(10),"\n\n")
#print (data.info())

#Plotting data received
#plt.style.use("classic")
#data["Close"].plot(label="TATA Motors", figsize=(16,8), title="Adjusted Closing Price")
#plt.legend()
#plt.show()

data["prediction"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = np.array(data.drop(["prediction"], 1))
Y = np.array(data["prediction"])

scale = StandardScaler()
X = scale.fit_transform(X)
X_prediction = X[-forecast_time:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

clf = LinearRegression()
clf.fit(X_train, Y_train)
prediction = (clf.predict(X_prediction))

print (prediction)
