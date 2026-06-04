# import pandas as pd
# from catboost import CatBoostRegressor

# # =========================
# # LOAD DATA
# # =========================

# train = pd.read_csv("../data/train.csv")
# test = pd.read_csv("../data/test.csv")

# # =========================
# # TIMESTAMP FEATURES
# # =========================

# train["timestamp"] = pd.to_datetime(train["timestamp"], format="%H:%M")
# test["timestamp"] = pd.to_datetime(test["timestamp"], format="%H:%M")

# for df in [train, test]:
#     df["hour"] = df["timestamp"].dt.hour
#     df["minute"] = df["timestamp"].dt.minute

#     df["is_morning_peak"] = (
#         (df["hour"] >= 7) & (df["hour"] <= 10)
#     ).astype(int)

#     df["is_evening_peak"] = (
#         (df["hour"] >= 16) & (df["hour"] <= 20)
#     ).astype(int)

# # =========================
# # GEOHASH FEATURES
# # =========================

# for df in [train, test]:
#     df["geo_1"] = df["geohash"].astype(str).str[0]
#     df["geo_2"] = df["geohash"].astype(str).str[:2]
#     df["geo_3"] = df["geohash"].astype(str).str[:3]

# # Drop original timestamp
# train.drop(columns=["timestamp"], inplace=True)
# test.drop(columns=["timestamp"], inplace=True)

# # =========================
# # FEATURES / TARGET
# # =========================

# y = train["demand"]
# X = train.drop("demand", axis=1)

# # =========================
# # HANDLE MISSING VALUES
# # =========================

# for col in X.columns:
#     if X[col].dtype == "object":
#         X[col] = X[col].fillna("Missing")
#     else:
#         X[col] = X[col].fillna(X[col].median())

# for col in test.columns:
#     if test[col].dtype == "object":
#         test[col] = test[col].fillna("Missing")
#     else:
#         test[col] = test[col].fillna(test[col].median())

# # =========================
# # CAT FEATURES
# # =========================

# cat_features = [
#     i for i, col in enumerate(X.columns)
#     if X[col].dtype == "object"
# ]

# print("Training V3 model...")

# # =========================
# # MODEL
# # =========================

# model = CatBoostRegressor(
#     iterations=2000,
#     learning_rate=0.03,
#     depth=10,
#     loss_function="RMSE",
#     random_seed=42,
#     verbose=200
# )

# # =========================
# # TRAIN
# # =========================

# model.fit(
#     X,
#     y,
#     cat_features=cat_features
# )

# # =========================
# # PREDICT
# # =========================

# predictions = model.predict(test)

# # =========================
# # SUBMISSION
# # =========================

# submission = pd.DataFrame({
#     "Index": test["Index"],
#     "demand": predictions
# })

# submission.to_csv(
#     "../submissions/submission_v3_geohash.csv",
#     index=False
# )

# print("\nSubmission created!")
# print(submission.head())
# print("\nShape:", submission.shape)
import pandas as pd

train = pd.read_csv("../data/train.csv")

print(train["timestamp"].head(20))
print(train["timestamp"].nunique())