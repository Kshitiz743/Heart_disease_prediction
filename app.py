from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(BASE_DIR, "encoders.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        gender = request.form["gender"]
        bp = float(request.form["blood_pressure"])
        cholesterol = float(request.form["cholesterol"])
        exercise = request.form["exercise"]
        smoking = request.form["smoking"]
        family = request.form["family_history"]
        diabetes = request.form["diabetes"]
        bmi = float(request.form["bmi"])
        high_bp = request.form["high_bp"]
        low_hdl = request.form["low_hdl"]
        high_ldl = request.form["high_ldl"]
        alcohol = request.form["alcohol"]
        stress = request.form["stress"]
        sleep = float(request.form["sleep"])
        sugar = request.form["sugar"]
        triglyceride = float(request.form["triglyceride"])
        fasting = float(request.form["fasting"])
        crp = float(request.form["crp"])
        homocysteine = float(request.form["homocysteine"])

        gender = encoders["Gender"].transform([gender])[0]

        exercise = encoders["Exercise Habits"].transform([exercise])[0]

        smoking = encoders["Smoking"].transform([smoking])[0]

        family = encoders["Family Heart Disease"].transform([family])[0]

        diabetes = encoders["Diabetes"].transform([diabetes])[0]

        high_bp = encoders["High Blood Pressure"].transform([high_bp])[0]

        low_hdl = encoders["Low HDL Cholesterol"].transform([low_hdl])[0]

        high_ldl = encoders["High LDL Cholesterol"].transform([high_ldl])[0]

        alcohol = encoders["Alcohol Consumption"].transform([alcohol])[0]

        stress = encoders["Stress Level"].transform([stress])[0]

        sugar = encoders["Sugar Consumption"].transform([sugar])[0]

        data = [[
            age,
            gender,
            bp,
            cholesterol,
            exercise,
            smoking,
            family,
            diabetes,
            bmi,
            high_bp,
            low_hdl,
            high_ldl,
            alcohol,
            stress,
            sleep,
            sugar,
            triglyceride,
            fasting,
            crp,
            homocysteine
        ]]

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][prediction] * 100

        if prediction == 1:

            result = "⚠ High Risk of Heart Disease"

            recommendation = """
            Please consult a cardiologist.
            Maintain a healthy diet.
            Exercise regularly.
            Avoid smoking and alcohol.
            """

        else:

            result = "✅ Low Risk of Heart Disease"

            recommendation = """
            Continue your healthy lifestyle.
            Exercise regularly.
            Eat nutritious food.
            Schedule routine health check-ups.
            """

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability, 2),
            recommendation=recommendation
        )

    except Exception as e:

        print(e)

        return render_template(
            "result.html",
            prediction="Invalid Input!",
            probability=0,
            recommendation="Please enter valid values."
        )


if __name__ == "__main__":
    app.run(debug=True)