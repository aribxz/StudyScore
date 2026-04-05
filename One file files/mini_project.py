from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

hours_studied = np.array([1, 2, 3, 4, 5, 6])
sleep_hours = np.array([8, 6, 7, 5, 9, 4])  # independent feature

x = np.column_stack((hours_studied, sleep_hours))
scores = np.array([40, 50, 65, 70, 80, 90])

noise = np.random.randn(len(scores)) * 0.5
scores_new = noise + scores

print(scores_new)
y = scores_new

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)
prediction2 = model.predict([[7,50]])
print(prediction2)

mse = mean_squared_error(y_test, y_prediction)

print("Hours coefficient:", model.coef_[0])
print("Sleep coefficient:", model.coef_[1])
print("MSE:", mse)

# # sort for smooth line
# sorted_idx = np.argsort(x.flatten())
# x_sorted = x[sorted_idx]

# # plot
# plt.figure(figsize=(8, 5))

# # scatter (actual data)
# plt.scatter(x, y, s=80, label="Actual Data")

# # regression line
# plt.plot(x_sorted, model.predict(x_sorted), linewidth=2, label="Regression Line")

# # labels & styling
# plt.title("Student hours vs scores visualized", fontsize=14)
# plt.xlabel("Hours Studied")
# plt.ylabel("Scores")
# plt.legend()
# plt.grid()

# plt.show()