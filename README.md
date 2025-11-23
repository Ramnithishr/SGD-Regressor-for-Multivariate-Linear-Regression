# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.

2.Standardize X and Y using StandardScaler for consistent scaling across features.

3.Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.

4.Train the model on the standardized training data.

5.Predict on the test data, inverse-transform predictions, compute mean squared error, and print results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Ramnithish R
RegisterNumber:  212224230219
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

df.info()

X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y=df[['AveOccup','HousingPrice']]
Y.info()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print(Y_pred)

```
## Output:
# Head

<img width="827" height="322" alt="425470532-ad9cac19-0c53-4d5c-a1be-5bfde6cf95c6" src="https://github.com/user-attachments/assets/1331da0c-cb7c-4eee-a4f3-e92b0c25da87" />


# Dataset info

<img width="521" height="396" alt="425470555-910dd336-e517-4537-b5d3-48db7514fd8f" src="https://github.com/user-attachments/assets/8256d0a9-2424-4d85-a924-e2ca37da19e7" />


# Removing columns

<img width="430" height="334" alt="425474425-fe28a0cd-3b96-4c0e-a1f9-2b2e5d8dbbea" src="https://github.com/user-attachments/assets/348ead49-ecf3-4b6d-b412-dcf871d9732b" />


# Columns info

<img width="502" height="224" alt="425470601-f7b5fb9d-3b9e-4374-b9f2-322e9c3626fd" src="https://github.com/user-attachments/assets/2c5634aa-2d39-479c-927f-bbd204ceae7a" />


# X_train,X_test,Y_train,Y_test

<img width="732" height="959" alt="425470641-e8c84223-7a5a-41dd-a4b0-93abed0d62ed" src="https://github.com/user-attachments/assets/f27b7dce-fc66-406b-8195-5d9b9eb46190" />


# MultiOutputRegressor(sgd)

<img width="262" height="189" alt="425540560-0b19f675-30d4-4d60-8e26-dd866d1ca975" src="https://github.com/user-attachments/assets/e321d796-88c3-4396-82d6-5ef98f246133" />


# Mean Squared error

<img width="429" height="196" alt="425470696-06fe8bd0-7130-45d1-8788-cae333292a09" src="https://github.com/user-attachments/assets/b41244ae-08c9-4ebf-b27b-a2baf9703ffd" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
