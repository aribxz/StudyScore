import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

np.random.seed(42)
n_samples = 200

hours_studied = np.random.uniform(0, 14, n_samples)
sleep_hours = np.random.uniform(4, 12, n_samples)
practice_tests = np.random.randint(0, 20, n_samples)
phone_usage = np.random.uniform(0, 10, n_samples)

sleep_bad = []
sleep_good = []

for s in sleep_hours:
    if (s < 6):
        sleep_bad.append(1)
        sleep_good.append(0)

    elif (6 <= s <= 8):
        sleep_bad.append(0)
        sleep_good.append(1)

    else:
        sleep_bad.append(0)
        sleep_good.append(0)    

sleep_bad = np.array(sleep_bad)
sleep_good = np.array(sleep_good)

study_effect = []
for s in range(n_samples): 
    if (sleep_bad[s]==1):
        study_effect.append(2* hours_studied[s])

    elif (sleep_good[s]==1):
        study_effect.append(5* hours_studied[s])

    else:
        study_effect.append(4* hours_studied[s])

study_effect = np.array(study_effect)

Hours_Sleep_Bad = hours_studied * sleep_bad
Hours_Sleep_Good = hours_studied * sleep_good

noise = np.random.normal(0, 5, n_samples)

# Formula :
score = (
    study_effect +
    3 * practice_tests -
    4 * phone_usage +
    (-10 * sleep_bad) +
    (10 * sleep_good) +
    noise
)

data = pd.DataFrame({
    "Hours Studied" : hours_studied,
    "Practice Tests" : practice_tests,
    "Sleep Hours" : sleep_hours,
    "Phone Usage" : phone_usage,
    "Sleep Bad" : sleep_bad,
    "Sleep Good" : sleep_good, 
    "Hours Sleep Bad" : Hours_Sleep_Bad,
    "Hours Sleep Good" : Hours_Sleep_Good,
    "Score" : score
})

print(data.head())

x = data[[
    "Hours Studied",
    "Practice Tests",
    "Sleep Bad", "Sleep Good",
    "Hours Sleep Bad", "Hours Sleep Good",
    "Phone Usage"
]]

y = data["Score"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)
mse = mean_squared_error(y_test, y_prediction)
print("Coefficient : ", model.coef_)
print("Intercept : ", model.intercept_)
print("MSE:", mse)

r2 = r2_score(y_test, y_prediction)
rmse  = np.sqrt(mse)

extras = {
    "mse"       : mse,
    "rmse"      : rmse,
    "r2"        : r2,
    "X_train"   : np.array(x_train),
    "n_train"   : len(x_train),
    "n_features": x_train.shape[1],
    "y_test"    : list(y_test),          # actual scores from test set
    "y_pred"    : list(y_prediction),    # what model predicted for those
    "feature_names": [
        "Hours Studied", "Practice Tests",
        "Sleep Bad", "Sleep Good",
        "Hours*Sleep Bad", "Hours*Sleep Good",
        "Phone Usage"
    ]
}

with open("extras.pkl", "wb") as f:
    pickle.dump(extras, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)