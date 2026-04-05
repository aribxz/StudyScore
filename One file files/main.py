from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(x,y)

print(model.predict([[5]]))
print(model.coef_)     # slope (m)
print(model.intercept_)  # intercept (b) 