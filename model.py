import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("heart_disease.csv")

print("\nDataset Loaded Successfully")
print(df.shape)

# =====================================================
# REPLACE BLANKS WITH NaN
# =====================================================

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# =====================================================
# TARGET COLUMN
# =====================================================

target = "Heart Disease Status"

# =====================================================
# SEPARATE FEATURES
# =====================================================

X = df.drop(target, axis=1)
y = df[target]

# =====================================================
# FIND COLUMNS
# =====================================================

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical Columns")
print(cat_cols)

print("\nNumerical Columns")
print(num_cols)

# =====================================================
# IMPUTE MISSING VALUES
# =====================================================

cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X[num_cols] = num_imputer.fit_transform(X[num_cols])

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

encoders = {}

for col in cat_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col])

    encoders[col] = le

# =====================================================
# ENCODE TARGET
# =====================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

encoders[target] = target_encoder

# Save encoders

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =====================================================
# TRAIN MODEL
# =====================================================

model = LogisticRegression(max_iter=3000)

model.fit(X_train, y_train)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# =====================================================
# EVALUATE MODEL
# =====================================================

pred = model.predict(X_test)

print("\nAccuracy")

print(f"{accuracy_score(y_test,pred)*100:.2f}%")

print("\nConfusion Matrix")

print(confusion_matrix(y_test,pred))

print("\nClassification Report")

print(classification_report(y_test,pred))

print("\nModel Saved Successfully")