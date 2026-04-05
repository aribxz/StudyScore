from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

data = load_iris()
x = data.data
y = data.target

# print(data.feature_names)
# print(data.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(max_depth=4)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))

print("Predictions:", y_pred[:10])
print("Actual:", y_test[:10])

pl.figure(figsize = (12,8))
plot_tree(model, filled=True)
plot_tree(model, filled=True, feature_names=data.feature_names)
pl.show()