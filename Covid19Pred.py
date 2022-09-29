# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 00:21:33 2021

@author: Abhinav

Abhinav
B20271
8306808320
b20271@students.iitmandi.ac.in
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.stats import pearsonr
from statsmodels.graphics import tsaplots
from statsmodels.tsa.ar_model import AutoReg as AR
import math
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.utils import check_array

#creating a MAPE function , It will be used later
'''
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
'''
#reading csv file
df=pd.read_csv("daily_covid_cases.csv")


#create a datetime object from given string
date_objects = [datetime.strptime(date, '%Y-%m-%d').date() for date in df['Date']]

#plotting line plot with x-axis as index of the day and y-axis as the number of Covid-19 cases
plt.plot(date_objects,df['new_cases'])
plt.xticks(rotation=45)
plt.xlabel("Month-Year")
plt.ylabel("No of Confirmed Cases")
plt.title("Covid Cases vs Time")
plt.show()




#creating a lagged timeseries
df_lag=df.shift(-1)

#plotting 
plt.scatter(df_lag['new_cases'],df['new_cases'],s=5)
plt.xlabel("Original time series")
plt.ylabel("Lagged time series")
plt.title("Original series vs time lagged series")
plt.show()

#calculatig pearson's coefficient
corrs=[]
def corr():
    for i in range(1,7):
        df_lag=df.shift(-i)
        corr = df['new_cases'].corr(df_lag['new_cases'])
        print(f'Pearsons correlation with lag {i} time series is', corr)
        corrs.append(corr)
corr()

#creating line plot between obtained correlation coefficients (on the y-axis) and lagged values (on the x-axis)

x=[1,2,3,4,5,6]
plt.plot(x,corrs)
#creating another plot to highlight the values
plt.plot(x,corrs,'ro')
#showing values also 
for a,b in zip(x, corrs): 
    plt.text(a, b, str(b))
plt.xlabel("Lag Values")
plt.ylabel("Correlation Values ")
plt.title(" Correlation coefficient vs. lags in given sequence")
plt.show()


#Plotting correlogram or Auto Correlation Function
fig = tsaplots.plot_acf(df['new_cases'], lags=50)
plt.xlabel("Lag Values")
plt.ylabel("Correlation Values ")
plt.title("Correlogram or Auto Correlation Function")
plt.show()

print("-------------------------------Q2-------------------------------------")
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

pd.DataFrame(train,columns=["new_cases"]).to_csv('train_saved.csv')


Window = 5 # The lag=5
model = AR(train, lags=Window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params 

print("coefficients from the trained AR model are:",coef)

Window = 5 # The lag=5
model = AR(train, lags=Window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params 

history = train[len(train)-Window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-Window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(Window):
        yhat += coef[d+1] * lag[Window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs)
    # Append actual test value to history, to be used in next step.   

#slicing first 5 data values from history
history=history[:len(predictions)]




#Plotting scatter plot
plt.scatter(test,predictions,s=5)
plt.xlabel("Actual Data")
plt.ylabel("Predicted values")
plt.title("Actual values vs predicted values")
plt.show()


#Plotting line plot
x=[i for i in range(len(test))]
plt.plot(x,test,color='r')
plt.plot(x,predictions,color='y')
plt.title("Predicted test data time sequence vs. original test data sequence")
plt.legend(['Actual','Predicted'])
plt.show()

#Calculating RMSE
mse =mean_squared_error(test, predictions,squared=False)
rmse=mse/np.mean(history)
mape=mean_absolute_percentage_error(test, predictions)
print('RMSE % is ',rmse*100)
print('Mape is ',mape)


print("---------------------------------------Q3------------------------------")

#creating a function 

def Autoref(p):
    
    
    Window = p # The lag=5
    model = AR(train, lags=Window) 
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params 
    
    history = train[len(train)-Window:]
    history = [history[i] for i in range(len(history))]
    
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-Window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(Window):
            yhat += coef[d+1] * lag[Window-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs)
    #slicing history 
    history=history[:len(predictions)]
    # Append actual test value to history, to be used in next step
    mse =mean_squared_error(test, predictions,squared=False)
    rmse=mse/np.mean(history)
    mape=mean_absolute_percentage_error(test, predictions)
    
    return rmse*100,mape
    
t=[1,5,10,15,25]

rmse_lst=[]
mape_lst=[]

for i in t:
    rmse_lst.append(Autoref(i)[0])
    mape_lst.append(Autoref(i)[1])
    print(f"RMSE AND MAPE for p={i} is", Autoref(i)[0],Autoref(i)[1])
    
plt.bar(t,rmse_lst)
plt.xlabel("Value of lag")
plt.ylabel("Value of RMSE %")
plt.title("Lag Value vs RMSE")
plt.show()

plt.bar(t,mape_lst,color='y')
plt.xlabel("Value of lag")
plt.ylabel("Value of MAPE")
plt.title("Lag Value vs MAPE")
plt.show()

 
print("---------------------------------------Q4------------------------------")


i=0
corr=1

#df_lag=df.shift(-i)
#corr = df['new_cases'].corr(df_lag['new_cases'])
train_saved=pd.read_csv("train_saved.csv")

while corr>2/(len(train))**0.5:
    i=i+1;
    train_lag=train_saved.shift(i)
    corr=train_saved['new_cases'].corr(train_lag['new_cases'])
    
print('Optimal Value of Lag is',i-1)



print(f'RMSE % and MAPE values at p={i-1} is',Autoref(i-1))



























