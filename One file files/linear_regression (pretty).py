from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

x, y = make_regression(
    n_samples=120,
    n_features=1,
    noise=20,
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()

# fit()     → learn pattern from training data  
model.fit(x_train, y_train)

# predict() → apply learned pattern to new data  
y_prediction = model.predict(x_test)

# MSE → measure how wrong predictions are  
mse = mean_squared_error(y_test, y_prediction)

print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print("MSE:", mse)

sorted_idx = np.argsort(x.flatten())

plt.scatter(x, y)
plt.plot(x[sorted_idx], model.predict(x)[sorted_idx])
plt.title("Linear Regression on Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



