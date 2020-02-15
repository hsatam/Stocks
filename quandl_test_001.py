# Import Libraries
import quandl
import numpy as numpy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

data = getQuandlInfo(STOCK, START_DATE, END_DATE, RETURNS)

#print info of data returned
print (data.head(10),"\n\n")
print (data.info())

#Plotting data received
plt.style.use("classic")
data["Close"].plot(label="TATA Motors", figsize=(16,8), title="Adjusted Closing Price")
plt.legend()
plt.show()

