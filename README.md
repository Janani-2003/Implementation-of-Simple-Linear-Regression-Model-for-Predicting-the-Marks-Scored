# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JANANI R
RegisterNumber: 212221230039

import numpy as np
import pandas as pd
dataset=pd.read_csv('/content/student_scores.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='orange')
plt.title('Training set (H vs S)')
plt.xlabel('Hours')
plt.ylabel("scores")
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,reg.predict(X_test),color='black')
plt.title('Test set (H vs S)')
plt.xlabel('Hours')
plt.ylabel("scores")
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE =  ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
