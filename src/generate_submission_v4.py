import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# =========================
# BETTER TIMESTAMP FEATURES
# =========================

for df in [train, test]:

    parts = df["timestamp"].str.split(":", expand=True)

    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)

    # 96 slots per day (15-minute intervals)
    df["time_slot"] = df["hour"] * 4 + (df["minute"] // 15)

    # Cyclical time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Peak traffic indicators
    df["is_morning_peak"] = (
        (df["hour"] >= 7) & (df["hour"] <= 10)
    ).astype(int)

    df["is_evening_peak"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20)
    ).astype(int)

# Drop original timestamp
train.drop(columns=["timestamp"], inplace=True)
test.drop(columns=["timestamp"], inplace=True)

# =========================
# TARGET
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

print("Training V4 model...")

# =========================
# MODEL
# =========================

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=10,
    loss_function="RMSE",
    random_seed=42,
    verbose=200
)

# =========================
# TRAIN
# =========================

model.fit(
    X,
    y,
    cat_features=cat_features
)

# =========================
# PREDICT
# =========================

predictions = model.predict(test)

# =========================
# SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": predictions
})

submission.to_csv(
    "../submissions/submission_v4_timeslot.csv",
    index=False
)

print("\nSubmission created!")
print(submission.head())
print("\nShape:", submission.shape)