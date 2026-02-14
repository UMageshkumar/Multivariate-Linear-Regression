# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
Step1
import pandas as pd.

Step2
Read the csv file.

Step3
Get the value of X and y variables

Step4
Create the linear regression model and fit.

Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.

## Program:
```
# Developed by: Mageshkumar U
# Register number: 212224240085

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# load California housing dataset
california = datasets.fetch_california_housing()

# defining feature matrix (X) and response vector (y)
X = california.data
y = california.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print("Coefficients:", reg.coef_)

# variance score
print("Variance score:", reg.score(X_test, y_test))

# plot style
plt.style.use('fivethirtyeight')

# training residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data"
)

# testing residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data"
)

# zero residual line
plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)

plt.legend(loc="upper right")
plt.title("Residual errors")
plt.show()





```
## Output:
<img width="775" height="663" alt="image" src="https://github.com/user-attachments/assets/6ac3e4c5-80fe-4850-824a-7e87e75f8cb1" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
