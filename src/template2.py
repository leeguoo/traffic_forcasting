import pandas as pd

#read raw data
raw = pd.read_csv('../../input/train_1.csv',index_col='Page',nrows=5000)
raw = raw.fillna(0)

#get target
df = pd.DataFrame()
df["traffic"] = raw.stack()

#get time-series features
for num in [62,63,64,65,66,67,68]:
    print(num)
    tag = "lag_{0}".format(num)
    df[tag] = raw.shift(num,axis=1).stack()
df = df.dropna(how="any")

#find feature and target names
features = list(df.columns.values)
target = 'traffic'
features.remove(target)

#split train and test
print("split")
train = df[df.index.get_level_values(1)<'2016-11-01']
test = df[df.index.get_level_values(1)>='2016-11-01']

train_X, train_y = train[features], train[target]
test_X, test_y = test[features], test[target]

print("model")
#random forest model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1)
rf.fit(train_X, train_y)
preds = rf.predict(test_X)

from smape import smape
print(smape(test_y,preds))
