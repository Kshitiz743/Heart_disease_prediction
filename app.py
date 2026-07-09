from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        if features[0] <= 0:
            return render_template(
                "result.html",
                prediction="Invalid Age",
                probability=0
            )

        data = scaler.transform([features])

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][prediction] * 100

        if prediction == 1:
            result = "⚠ High Risk of Heart Disease"
        else:
            result = "✅ Low Risk of Heart Disease"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability, 2)
        )

    except Exception as e:
        print(e)
        return render_template(
            "result.html",
            prediction="Invalid Input!",
            probability=0
        )

if __name__ == "__main__":
    app.run(debug=True)