# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MOHANRAJ R
RegisterNumber:  212224230169

*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print("df.head")

df.head()

print("df.tail")

df.tail()

Y=df.iloc[:,1].values
print("Array of Y")
Y

X=df.iloc[:,:-1].values
print("Array of X")
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Array values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

<img width="781" height="265" alt="image" src="https://github.com/user-attachments/assets/6956063b-81cc-4566-b61a-e5e65d4600b6" />

<img width="560" height="260" alt="image" src="https://github.com/user-attachments/assets/a213c8df-fb10-42e0-8f93-0fdace95d9b1" />


<img width="742" height="81" alt="image" src="https://github.com/user-attachments/assets/6fc829d4-9330-4bf4-93b7-7a2941b18fc1" />
<img width="278" height="583" alt="image" src="https://github.com/user-attachments/assets/87f49aa2-0999-404d-b4a7-7baf5cbfe8d0" />
<img width="1175" height="105" alt="image" src="https://github.com/user-attachments/assets/f26e53f7-7384-411c-99c9-c220bbe03947" />

<img width="589" height="76" alt="image" src="https://github.com/user-attachments/assets/51785ac2-8e9b-45f3-96ce-327c1b959d6f" />


<img width="813" height="620" alt="Screenshot 2025-08-28 091525" src="https://github.com/user-attachments/assets/c433c1c6-1a2d-4ffc-ba79-6ee59399b8ee" />

<img width="924" height="634" alt="image" src="https://github.com/user-attachments/assets/8810ed18-aeca-4c34-857e-85de17bed69a" />

<img width="1026" height="100" alt="image" src="https://github.com/user-attachments/assets/b345021f-1e75-4769-a588-bd8e0e423d3b" />




![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
