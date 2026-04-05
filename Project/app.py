from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from scipy import stats

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("extras.pkl", "rb") as f:
    extras = pickle.load(f)


@app.route("/")
def home():
    return render_template("index3.html")


# Feature builder
def get_features(hours, tests, sleep, phone):
        sleep_bad = 1 if sleep < 6 else 0
        sleep_good = 1 if 6 <= sleep <= 8 else 0

        Hours_sleep_bad = hours * sleep_bad
        Hours_sleep_good = hours * sleep_good

        return np.array([[ 
            hours,
            tests,
            sleep_bad,
            sleep_good,
            Hours_sleep_bad,
            Hours_sleep_good,
            phone
        ]])


@app.route("/predict", methods=["POST"])
def predict():

    # 🔹 Handle BOTH JSON and form requests
    if request.is_json:
        data = request.get_json()
        hours = float(data["hours"])
        tests = float(data["tests"])
        sleep = float(data["sleep"])
        phone = float(data["phone"])
    else:
        hours = float(request.form["hours"])
        tests = float(request.form["tests"])
        sleep = float(request.form["sleep"])
        phone = float(request.form["phone"])
    

    # 🔹 Base prediction
    base_score = np.clip(model.predict(get_features(hours, tests, sleep, phone))[0], 0, 100)

    margin = prediction_interval(get_features(hours, tests, sleep, phone))
    lower = round(np.clip(base_score - margin, 0, 100), 1)
    upper = round(np.clip(base_score + margin, 0, 100), 1)

    # 🔹 What-if simulations
    better_sleep_score = np.clip(model.predict(get_features(hours, tests, 7, phone))[0], 0, 100)
    low_phone_score    = np.clip(model.predict(get_features(hours, tests, sleep, 2))[0], 0, 100)
    more_study_score   = np.clip(model.predict(get_features(hours + 2, tests, sleep, phone))[0], 0, 100)

    # 🔹 Gains
    sleep_gain = better_sleep_score - base_score
    phone_gain = low_phone_score - base_score
    study_gain = more_study_score - base_score

    gains = {
        "sleep": sleep_gain,
        "phone": phone_gain,
        "study": study_gain
    }

    best = max(gains, key=gains.get)

    # 🔹 Insight
    if base_score==100:
        insight = f"Perfect! You have outdone yourself."

    elif best == "sleep":
        insight = f"Improving sleep could increase your score by {round(sleep_gain,1)}"

    elif best == "phone":
        insight = f"Reducing phone usage could increase your score by {round(phone_gain,1)}"
    else:
        insight = f"Studying 2 more hours could increase your score by {round(study_gain,1)}"

    # 🔹 Return EVERYTHING frontend needs
    return jsonify({
        "score": round(float(base_score), 2),
        "insight": insight,

        "graph": {
            "Current": base_score,
            "Better Sleep": better_sleep_score,
            "Less Phone": low_phone_score,
            "More Study": more_study_score
        },

        "interval": {
            "lower": lower,
            "upper": upper, 
            "margin": margin
        },

        "individual_graphs": {
            "hours": [
                model.predict(get_features(h, tests, sleep, phone))[0]
                for h in range(1, 11)
            ],
            "tests": [
                model.predict(get_features(hours, t, sleep, phone))[0]
                for t in range(1, 16)
            ],
            "sleep": [
                model.predict(get_features(hours, tests, s, phone))[0]
                for s in range(3, 11)
            ],
            "phone": [
                model.predict(get_features(hours, tests, sleep, p))[0]
                for p in range(1, 11)
            ]
        }
    })


def prediction_interval(x_input, confidence=0.95):
    mse = extras["mse"]
    X_train = extras["X_train"]
    n = extras["n_train"]

    # XᵀX inverse — measures spread of training data
    XtX_inv = np.linalg.inv(X_train.T @ X_train)

    # How "unusual" is this input compared to training data
    leverage = (x_input @ XtX_inv @ x_input.T).item()

    # t-value for 95% confidence
    t = stats.t.ppf((1 + confidence) / 2, df=n - extras["n_features"] - 1)

    # Final margin
    margin = t * np.sqrt(mse * (1 + leverage))
    return round(margin, 1)

@app.route("/goal", methods=["POST"])
def goal():
    data = request.get_json()
    hours = float(data["hours"])
    tests = float(data["tests"])
    sleep = float(data["sleep"])
    phone = float(data["phone"])
    target = float(data["target"])

    base_score = model.predict(get_features(hours, tests, sleep, phone))[0]
    gap = target - base_score
    coef = model.coef_

    suggestions = []
    # coef[0]=hours, [1]=tests, [2]=sleep_bad, [3]=sleep_good,
    # [4]=hours_sleep_bad, [5]=hours_sleep_good, [6]=phone

    # ── 1. PHONE ─────────────────────────────────────────────
    # phone only appears once, coefficient is coef[6]
    phone_coef = coef[6]
    needed_phone = phone + gap / phone_coef
    needed_phone = round(np.clip(needed_phone, 0, 10), 1)
    suggestions.append({
        "variable" : "Phone Usage",
        "current" : phone,
        "needed" : needed_phone,
        "unit" : "hrs/day",
        "feasible" : bool(0 <= needed_phone <= 10)
    })

    # ── 2. HOURS ─────────────────────────────────────────────
    # hours appears in 3 features, effective coef depends on sleep
    sleep_bad_flag = 1 if sleep < 6 else 0 
    sleep_good_flag = 1 if 6 <= sleep <=8 else 0
    hours_coef = coef[0] + coef[4] * sleep_bad_flag + coef[5] * sleep_good_flag
    needed_hours = hours + gap / hours_coef
    needed_hours = round(np.clip(needed_hours, 0, 14), 1) 
    suggestions.append({
        "variable" : "Hours",
        "current" : hours,
        "needed" : needed_hours,
        "unit" : "hrs/day",
        "feasible" : bool(0 <= needed_hours <= 14)
    })

    # ── 3. TESTS ─────────────────────────────────────────────
    tests_coef = coef[1]
    needed_tests = tests + gap / tests_coef
    needed_tests = round(np.clip(needed_tests, 0, 20), 1)
    suggestions.append({
        "variable" : "Tests",
        "current" : tests,
        "needed" : needed_tests,
        "unit" : "tests",
        "feasible" : bool(0 <= needed_tests <= 20)
    })

    return jsonify({
        "current_score": round(float(base_score), 1),
        "target": target,
        "gap": round(float(gap), 1),
        "suggestions": suggestions
    })

@app.route("/report", methods=["GET"])
def report():
    coef = model.coef_
    intercept = model.intercept_

    feature_importance = []

    for name, c in zip(extras["feature_names"], coef):
        feature_importance.append({
            "feature" : name,
            "coefficient" : round(float(c), 3)
        })

    feature_importance.sort(key = lambda x : abs(x["coefficient"]), reverse=True)

    return jsonify({
        "r2"       : round(float(extras["r2"]), 4),
        "mse"      : round(float(extras["mse"]), 2),
        "rmse"     : round(float(extras["rmse"]), 2),
        "intercept": round(float(intercept), 2),
        "n_train"  : extras["n_train"],
        "n_test"   : len(extras["y_test"]),
        "feature_importance": feature_importance,
        "scatter": {
            "actual"   : [round(v, 1) for v in extras["y_test"]],
            "predicted": [round(v, 1) for v in extras["y_pred"]]
        }
    })

if __name__ == "__main__":
    app.run(debug=True)