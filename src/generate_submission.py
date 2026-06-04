import pandas as pd
from catboost import CatBoostRegressor

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# =========================
# FEATURES / TARGET
# =========================

y = train["demand"]
X = train.drop("demand", axis=1)

# =========================
# HANDLE MISSING VALUES
# =========================

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Missing")
    else:
        X[col] = X[col].fillna(X[col].median())

for col in test.columns:
    if test[col].dtype == "object":
        test[col] = test[col].fillna("Missing")
    else:
        test[col] = test[col].fillna(test[col].median())

# =========================
# CATEGORICAL FEATURES
# =========================

cat_features = [
    i for i, col in enumerate(X.columns)
    if X[col].dtype == "object"
]

print("Training model...")

# =========================
# MODEL
# =========================

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)

# =========================
# TRAIN ON FULL TRAIN DATA
# =========================

model.fit(
    X,
    y,
    cat_features=cat_features
)

# =========================
# PREDICT TEST DATA
# =========================

print("Generating predictions...")

predictions = model.predict(test)

# =========================
# CREATE SUBMISSION FILE
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": predictions
})

print("\nSubmission Preview:")
print(submission.head())

print("\nSubmission Shape:")
print(submission.shape)

submission.to_csv(
    "../submissions/submission_catboost.csv",
    index=False
)

print("\n✅ submission_catboost.csv created successfully!")
print("Location: ../submissions/submission_catboost.csv")