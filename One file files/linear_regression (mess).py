from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

x = np.array([[1], [2], [3], [4]])
y = np.array([4, 0, 69, 99])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
prediction2 = model.predict([[5]])
print(prediction2)

mse = mean_squared_error(y_test, prediction)

print(model.coef_)      # slope (m)
print(model.intercept_) # intercept (b)
print(mse)

pl.scatter(x, y)
pl.plot(x, model.predict(x))
pl.show()

model = DecisionTreeRegressor()
model.fit(x_train, y_train)

pred = model.predict(x)

pl.scatter(x, y)
pl.plot(x, pred)
pl.show()