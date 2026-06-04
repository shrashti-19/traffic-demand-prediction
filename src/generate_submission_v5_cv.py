import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# =========================
# TIMESTAMP FEATURES (V4)
# =========================

for df in [train, test]:

    parts = df["timestamp"].str.split(":", expand=True)

    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)

    df["time_slot"] = df["hour"] * 4 + (df["minute"] // 15)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_morning_peak"] = (
        (df["hour"] >= 7) & (df["hour"] <= 10)
    ).astype(int)

    df["is_evening_peak"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20)
    ).astype(int)

train.drop(columns=["timestamp"], inplace=True)
test.drop(columns=["timestamp"], inplace=True)

# =========================
# TARGET
# =========================

y = train["demand"]
X = train.drop("demand", axis=1)

# =========================
# MISSING VALUES
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
# CAT FEATURES
# =========================

cat_features = [
    i for i, col in enumerate(X.columns)
    if X[col].dtype == "object"
]

# =========================
# 5-FOLD CV
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

test_predictions = np.zeros(len(test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

    print(f"\n========== Fold {fold+1} ==========")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=10,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    fold_preds = model.predict(test)

    test_predictions += fold_preds / 5

# =========================
# SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": test_predictions
})

submission.to_csv(
    "../submissions/submission_v5_cv.csv",
    index=False
)

print("\n✅ V5 submission created!")
print(submission.head())
print("\nShape:", submission.shape)