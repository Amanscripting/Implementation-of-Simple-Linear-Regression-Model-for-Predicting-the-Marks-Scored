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
6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```

Developed by: Aman Alam
RegisterNumber:  212224240011
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv('/content/student_scores.csv')

#displaying the content in datafile.

df.head()

df.tail()

X=df.iloc[:,:-1].values

X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split

X_train,X_test, Y_train,Y_test=train_test_split(X, Y, test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train, Y_train)

Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train, Y_train,color="orange")

plt.plot(X_train, regressor.predict(X_train),color="red")

plt.title("Hours vs Scores (Training Set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

plt.scatter(X_test, Y_test,color="purple")

plt.plot(X_test, regressor.predict(X_test), color="yellow")

plt.title("Hours vs Scores (Test Set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

mse=mean_squared_error(Y_test, Y_pred)

print('MSE = ',mse)

mae=mean_absolute_error(Y_test, Y_pred)

print('MAE = ', mae)

rmse=np.sqrt(mse)

print("RMSE = ",rmse)
```
## Output:
```
![Screenshot 2025-04-20 124318](https://github.com/user-attachments/assets/747e6f48-426a-45f4-9062-ad618c0d5aa5)

![Screenshot 2025-04-20 163002](https://github.com/user-attachments/assets/e827d648-1eb9-4705-ad3e-ab43393d65d1)

![Screenshot 2025-04-20 163014](https://github.com/user-attachments/assets/a9983c3d-8fd4-473a-bf0f-36dfc7c4e15b)

![Screenshot 2025-04-20 163056](https://github.com/user-attachments/assets/dbe3560f-5a92-4c68-a7dc-de7b0947f330)

![Screenshot 2025-04-20 163123](https://github.com/user-attachments/assets/daf973e7-0784-4ae6-8b6b-f7baa4a8f2f9)

![Screenshot 2025-04-20 163146](https://github.com/user-attachments/assets/73e3dfb1-bfd7-41a1-bdc4-049cfcb4db48)

![Screenshot 2025-04-20 163203](https://github.com/user-attachments/assets/af907c17-fd99-4c0c-83d6-626ba582f8d9)

![Screenshot 2025-04-20 163219](https://github.com/user-attachments/assets/404fc74c-b7d6-4e63-aaf1-9010f8617b7e)

![Screenshot 2025-04-20 163234](https://github.com/user-attachments/assets/7d105bc1-45cf-44d1-9ebf-84b53179fe39)
```
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
