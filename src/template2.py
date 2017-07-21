import pandas as pd

#read raw data
raw = pd.read_csv('../../input/train_1.csv',index_col='Page',nrows=10000)
raw = raw.fillna(0)

#get target
df = pd.DataFrame()
df["traffic"] = raw.stack()

#get time-series features
weekmedian = raw.rolling(7,axis=1).median()
biweekaverage = weekmedian.rolling(14,axis=1).mean()
monthaverage = weekmedian.rolling(28,axis=1).mean()
for num in [61,91,121,151,181]:
    print(num)
    tag = "lag_{0}".format(num)
    df[tag] = raw.shift(num,axis=1).stack()
    df["WKM_"+tag] = weekmedian.shift(num,axis=1).stack()
    df["BWKA_"+tag] = biweekaverage.shift(num,axis=1).stack()
    df["MA_"+tag] = monthaverage.shift(num,axis=1).stack()
    

#df["weekmedian"] = raw.rolling(7,axis=1).median().shift(61,axis=1).stack()

df = df.dropna(how="any")

#print df[["traffic","WKM_lag_61","WKM_lag_91","WKM_lag_121"]]

print df.iloc[1,:]
print df.iloc[2,:]

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
#from sklearn.ensemble import RandomForestRegressor
#
#rf = RandomForestRegressor(50,n_jobs=-1)
#rf.fit(train_X, train_y)
#preds = rf.predict(test_X)

#linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_X,train_y)
preds = lr.predict(test_X)

from smape import smape
print(smape(test_y,preds))
