# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JANANI R
RegisterNumber: 212221230039

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/student_scores - student_scores.csv")
df.head()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()898
*/
```

## Output:
![Screenshot 2022-12-01 204519](https://user-images.githubusercontent.com/94288340/205089949-bd04a6c9-8cef-4767-80dd-b8c6e819adf2.png)
![Screenshot 2022-12-01 204551](https://user-images.githubusercontent.com/94288340/205090005-a64e1e5c-85ab-4bdc-8cb6-cb19333f51b0.png)
![Screenshot 2022-12-01 204622](https://user-images.githubusercontent.com/94288340/205090045-a0d41554-e605-4416-8c34-f9a1d102c06f.png)
![Screenshot 2022-12-01 205454](https://user-images.githubusercontent.com/94288340/205091950-ebf45f6b-27b3-4c49-93d9-8f33b439d6bb.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
