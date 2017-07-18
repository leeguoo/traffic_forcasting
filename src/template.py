import pandas as pd

df = pd.read_csv('../input/train_1.csv',index_col='Page',nrows=1000)

df = df.stack().reset_index()

df["date"] = pd.to_datetime(df.level_1)
df["year"] = df.date.dt.year
df["month"] = df.date.dt.month
df["day"] = df.date.dt.day
df["weekday"] = df.date.dt.dayofweek
print(df)
