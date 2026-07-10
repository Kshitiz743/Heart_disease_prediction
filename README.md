# ❤️ Heart Disease Prediction System

A Machine Learning web application that predicts the likelihood of heart disease based on a patient's medical information. The application is built using **Python, Flask, Scikit-learn, HTML, CSS, and Bootstrap**.

## 📌 Features

* Predicts the risk of heart disease using a trained Machine Learning model.
* Simple and user-friendly web interface.
* Real-time prediction based on user input.
* Preprocessing using a saved StandardScaler.
* Fast and lightweight Flask backend.

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, Bootstrap
* **Backend:** Flask (Python)
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Pickle

## 📂 Project Structure

```
Heart-Disease-Prediction/
│
├── app.py
├── logistic_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── images/
│
└── dataset/
    └── heart.csv
```

## 📊 Dataset

The model is trained on a heart disease dataset containing various medical attributes such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression
* Slope of ST Segment
* Number of Major Vessels
* Thalassemia

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open your browser

```
http://127.0.0.1:5000
```

## 🚀 Machine Learning Model

* Algorithm: **Logistic Regression**
* Data Scaling: **StandardScaler**
* Model Storage: **Pickle (.pkl)**

## 📸 Application Workflow

1. Enter patient details.
2. Click the **Predict** button.
3. The application preprocesses the input.
4. The trained Logistic Regression model predicts the result.
5. The prediction is displayed on the webpage.

## 📈 Future Improvements

* Add multiple ML algorithms for comparison.
* Display prediction probability.
* Integrate feature importance visualization.
* User authentication and prediction history.
* Deploy on Render, Railway, or PythonAnywhere.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

## 📄 License

This project is intended for educational purposes.

## 👨‍💻 Author

**Kshitiz Gupta**

Computer Science Student | Machine Learning Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub!
