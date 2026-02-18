1️⃣ Choose a Binary Classification Dataset
Objective

Select a dataset with two output classes.

Dataset Used

Breast Cancer Wisconsin Dataset

Target column:

diagnosis

Classes:

M → Malignant (Cancer) → 1

B → Benign (No Cancer) → 0

Code
import pandas as pd

df = pd.read_csv("data.csv")

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print(df["diagnosis"].value_counts())
Result

Binary classification dataset confirmed.

2️⃣ Train-Test Split and Standardize Features
Objective

Split dataset and scale features for better model performance.

Code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Result

80% training data

20% testing data

Features standardized

3️⃣ Fit Logistic Regression Model
Objective

Train Logistic Regression model.

Code
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model trained successfully")
Result

Model learned relationship between features and diagnosis.

4️⃣ Evaluate Model Performance
Objective

Measure performance using evaluation metrics.

Code
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
Metric Explanation

Confusion Matrix
Shows correct and incorrect predictions

Precision
Measures accuracy of positive predictions

Recall
Measures ability to detect positive cases

ROC-AUC
Measures overall model performance

5️⃣ Tune Threshold and Explain Sigmoid Function
Objective

Adjust classification threshold and understand probability conversion.

Threshold Tuning Code
threshold = 0.3

y_pred_custom = (y_prob >= threshold).astype(int)

print(confusion_matrix(y_test, y_pred_custom))

Lower threshold → Higher recall
Higher threshold → Higher precision

Sigmoid Function

Formula:

[
σ(z) = 1 / (1 + e^{-z})
]

Purpose:

Converts model output into probability between 0 and 1.

Example:

Output = 0.8 → Cancer
Output = 0.2 → No Cancer
