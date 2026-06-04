import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("../data/train.csv")

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

# =========================
# CATEGORICAL FEATURES
# =========================

cat_features = [
    i for i, col in enumerate(X.columns)
    if X[col].dtype == "object"
]

print("Categorical feature indices:")
print(cat_features)

# =========================
# TRAIN / VALID SPLIT
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# MODEL
# =========================

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=100
)

# =========================
# TRAIN
# =========================

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

# =========================
# VALIDATION
# =========================

preds = model.predict(X_valid)

score = r2_score(y_valid, preds)

print("\n======================")
print("Validation R2:", score)
print("======================")