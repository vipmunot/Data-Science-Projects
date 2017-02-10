import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
stocks = pd.read_csv("sphist.csv")
stocks[["Date"]] = pd.to_datetime(stocks["Date"])
stocks.sort(columns = "Date",axis = 0, ascending = True, inplace = True)
print(stocks.head())
# Finding the close values.
close_values = []
for stock in stocks.iterrows():
    close_values.append(stock[1]["Close"])

def aggregate_timeseries(values_list, days, func):
    append_tolist = list(np.zeros((days,), dtype=np.int))
    j = None
    for i in range(len(values_list)):
        if i >= days:
            if i-days != 0:
                j = i-days
            append_tolist.append(func(values_list[i:j:-1]))
    return append_tolist

            
stocks["Close_5"] = aggregate_timeseries(close_values, 5, np.mean)
stocks["Close_30"] = aggregate_timeseries(close_values, 30, np.mean)
stocks["Std_5"] = aggregate_timeseries(close_values, 5, np.std)    

stocks = stocks[stocks["Close_30"] != 0]
stocks = stocks.dropna(axis = 0)

train = stocks[stocks["Date"] < datetime(year=2013, month =1 , day = 1)]
test = stocks[stocks["Date"] >= datetime(year=2013, month =1 , day = 1)]

print("Shape of training data:",train.shape)
print("Shape of test data",test.shape)

model = LinearRegression()
model.fit(train[["Close_5","Close_30","Std_5"]], train["Close"])
predict_close = model.predict(test[["Close_5","Close_30","Std_5"]])
actual_close = test["Close"]

mean_absolute_error = np.mean(abs(predict_close - actual_close))
print("The mean absolute error is ", mean_absolute_error)