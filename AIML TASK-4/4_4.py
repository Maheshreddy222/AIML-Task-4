
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Drop unnecessary columns
df = df.drop("id", axis=1)
df = df.drop("Unnamed: 32", axis=1)

# Step 3: Convert diagnosis to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Step 4: Split features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# Evaluation Metrics
# ===============================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)

# Print results
print("Confusion Matrix:")
print(cm)

print("\nPrecision:", precision)

print("\nRecall:", recall)

print("\nROC-AUC Score:", roc_auc)


# ===============================
# Plot ROC Curve
# ===============================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
