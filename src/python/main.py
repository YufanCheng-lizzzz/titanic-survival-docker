
import pandas as pd
from pathlib import Path

# Find the data folder
data_dir = Path("/app/data")

if not data_dir.exists():
    data_dir = Path(__file__).resolve().parent.parent / "data"

# Load the training data
train = pd.read_csv(data_dir / "train.csv")

# Basic data exploration
print("Training data shape:")
print(train.shape)

print("\nFirst five rows:")
print(train.head())

print("\nColumn information:")
train.info()

print("\nMissing values:")
print(train.isnull().sum())


# Explore survival rates
print("\nSurvival rate by sex:")
print(train.groupby("Sex")["Survived"].mean())

print("\nSurvival rate by passenger class:")
print(train.groupby("Pclass")["Survived"].mean())

# Select features for the model
features = ["Pclass", "Sex", "Age", "SibSp",
            "Parch", "Fare", "Embarked"]

X = train[features].copy()

# Handle missing values
age_median = train["Age"].median()
embarked_mode = train["Embarked"].mode()[0]

X["Age"] = X["Age"].fillna(age_median)
X["Embarked"] = X["Embarked"].fillna(embarked_mode)

# Convert categorical variables to numeric columns
X = pd.get_dummies(X, columns=["Sex", "Embarked"], dtype=int)

print("\nData after cleaning:")
print(X.head())

print("\nMissing values after cleaning:")
print(X.isnull().sum())


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Target variable
y = train["Survived"]

# Train logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Predict on training data
train_predictions = model.predict(X)

# Evaluate training accuracy
train_accuracy = accuracy_score(y, train_predictions)

print("\nLogistic Regression Results:")
print(f"Training accuracy: {train_accuracy:.4f}")


from sklearn.model_selection import train_test_split

# Split the original data before cleaning
X_raw = train[features].copy()
y = train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42,
    stratify=y
)

# Calculate missing value replacements using training data only
age_median = X_train["Age"].median()
embarked_mode = X_train["Embarked"].mode()[0]

for df in [X_train, X_val]:
    df["Age"] = df["Age"].fillna(age_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

# Encode categorical variables
X_train = pd.get_dummies(
    X_train, columns=["Sex", "Embarked"], dtype=int
)
X_val = pd.get_dummies(
    X_val, columns=["Sex", "Embarked"], dtype=int
)

# Match columns between training and validation
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# Train a new model using only the training split
validation_model = LogisticRegression(max_iter=2000)
validation_model.fit(X_train, y_train)

# Evaluate the model
train_pred = validation_model.predict(X_train)
val_pred = validation_model.predict(X_val)

print("\nModel Validation Results:")
print(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Held-out test accuracy (20% of train.csv): {accuracy_score(y_val, val_pred):.4f}")


# Load the Kaggle test dataset
test = pd.read_csv(data_dir / "test.csv")

print("\nTest data shape:")
print(test.shape)

print("\nMissing values in test data:")
print(test.isnull().sum())

# Prepare test features
X_test = test[features].copy()

# Fill missing values using the full training dataset
X_test["Age"] = X_test["Age"].fillna(train["Age"].median())
X_test["Embarked"] = X_test["Embarked"].fillna(
    train["Embarked"].mode()[0]
)
X_test["Fare"] = X_test["Fare"].fillna(train["Fare"].median())

# Convert categorical variables
X_test = pd.get_dummies(
    X_test, columns=["Sex", "Embarked"], dtype=int
)

# Match the training feature columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("\nMissing values after test data cleaning:")
print(X_test.isnull().sum())

# Predict using the model trained on all training data
test_predictions = model.predict(X_test)

results = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

print("\nFirst 10 test predictions:")
print(results.head(10))

print("\nPredicted survival counts:")
print(results["Survived"].value_counts())

# Check whether test accuracy can be calculated
if "Survived" in test.columns:
    test_accuracy = accuracy_score(
        test["Survived"], test_predictions
    )
    print(f"\nTest accuracy: {test_accuracy:.4f}")
else:
    print("\nTest accuracy cannot be calculated.")
    print("The Kaggle test.csv does not contain survival labels.")
    print("Model performance is evaluated using a held-out 20% split of train.csv.")