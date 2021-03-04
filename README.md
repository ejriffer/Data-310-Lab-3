# Question 1: An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.

True. An OLS model tries to find a model that predicts values as close to the actual dependent variables as possible.

# Question 2: Do you agree or disagree with the following statement: In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.

Disagree. Noise can be very hard to correlate with, and a good fit can be found in a linear model even if there is some noise in the data.

# Question 3: Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:

import pandas as pd

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LinearRegression

import numpy as np

from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/L3Data.csv')

X = df.drop('Grade',axis = 1)

y = df['Grade']

X_train, X_test, y_train, y_test = tts(X,y,test_size = 0.25, random_state = 1234)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print(rmse)

## Output = 5.8612799671644344

# Question 4: In practice we determine the weights for linear regression with the "X_test" data.

True. X_test is the test data that is run until a proper linear model can be determined including the best linear weights for the regression. 

# Question 5: Polynomial regression is best suited for functional relationships that are non-linear in weights.

False. Polynomial regression is best suited for functional relationships that are non-linear in the data, not the weights. 

# Question 6: Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

True. Linear Regression, multiple linear regression, and polynomail regression can be fit using LinearRegression(), for polynomial regression you need to use Polynomial Features first. 

# Question 7: Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is:

X_1 = X.drop('questions', axis = 1)

y = df['Grade']

x_train,x_test,y_train,y_test = tts(X_1,y, test_size = 0.25, random_state = 1234)

len(x_train)

## Output = 23

# Question 8: The gradient descent method does not need any hyperparameters.

False. The gradient descent method needs hyperparameters such as learning_rate, initial_n, initial_m and num_iterations.

# Question 9: To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?

#1

import matplotlib.pyplot as plt

#2

fig, ax = plt.subplots()

#3

ax.scatter(X_test, y_test, color="black", label="Truth")

ax.scatter(X_test, lin_reg.predict(X_test), color="green", label="Linear")

ax.set_xlabel("Discussion Contributions")

ax.set_ylabel("Grade")

#4

ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)

ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)

ax.minorticks_on()

# Question 10: Which of the following forms is not  linear in the weights ?

D. D is not linear in weights becaue it is in the exponent of e and is taken to the fourth power. 
