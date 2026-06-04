import pandas as pd

train = pd.read_csv("data/train.csv")

print("\nShape:")
print(train.shape)

print("\nColumns:")
print(train.columns)

print("\nData Types:")
print(train.dtypes)

print("\nMissing Values:")
print(train.isnull().sum())