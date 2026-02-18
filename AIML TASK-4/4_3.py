# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Drop unnecessary columns
df = df.drop("id", axis=1)
df = df.drop("Unnamed: 32", axis=1)

# Step 3: Convert target column to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Step 4: Separate features and target
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

# Step 7: Create Logistic Regression model
model = LogisticRegression()

# Step 8: Fit (train) the model
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")
