# Importing Libraries for data Exploration and Visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# importing daya from csv file
dataframe = pd.read_csv('weatherHistory.csv')
df = dataframe.copy()  # making a copy of data
print(df.head())  # checking head of data for initial understanding of data
print(df.shape)  # checking data shape
print(df.dtypes)  # checking column types
print(df.isnull().sum())  # checking null values in columns

df.fillna('rain', inplace=True)  # fill missing values in Precip Type column

# ------ Visualization ---------
# creating simple regression plots to see the behaviour of dependent and independent variables
plt.figure(figsize=(10,6))
sns.regplot(x='Humidity', y='Temperature (C)', data=df, lowess=True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='Humidity', y='Apparent Temperature (C)', data=df, lowess=True)
plt.show()

# Separating independent features and Apparent Temperature
X = df[['Humidity','Temperature (C)','Wind Speed (km/h)','Wind Bearing (degrees)','Pressure (millibars)','Loud Cover','Visibility (km)']]
Y = df[['Apparent Temperature (C)']]

# ---------- Normalizing data --------
# we normalize data so that all features have values in same range so that no feature can dictate the learning algorithm
scaler = StandardScaler()
scaler.fit(X)
Xfit = scaler.transform(X)
X = pd.DataFrame(data = Xfit, columns=X.columns)
print(X.head())

# --------- Splitting train and test data --------
xtrain, xtest, ytrain, ytest = train_test_split(X.values, Y.values, test_size = 0.18)  # you can vary the split size
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

# -------- Modeling -----------
# using simple linear regression algorithm
model = LinearRegression(fit_intercept=True, n_jobs =1)
model.fit(xtrain, ytrain)  # training the model
print(model.coef_)  # checking weights

pred = model.predict(xtest)  # making a prediction on test set

# comparing predicted values with original values
print(pred[:5])
print(ytest[:5])