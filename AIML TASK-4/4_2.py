# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Drop unnecessary columns
df = df.drop("id", axis=1)
df = df.drop("Unnamed: 32", axis=1)

# Step 3: Convert target column into numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Step 4: Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 6: Standardize features
scaler = StandardScaler()

# Fit on training data and transform
X_train = scaler.fit_transform(X_train)

# Transform test data
X_test = scaler.transform(X_test)

print("Features standardized successfully!")
