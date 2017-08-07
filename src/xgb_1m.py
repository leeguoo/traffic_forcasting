import pandas as pd

#read raw data

class WebTraffic(object):
    def __init__(self):
        pass

    def ReadTrain(self,trainpath):
        self.raw = pd.read_csv(trainpath,index_col='Page',nrows=5000)
        self.raw = self.raw.fillna(0)

        self.df = pd.DataFrame()
        self.df["traffic"] = self.raw.stack()

#    def WKMediaLag(self, lags=[61,91,121,151,181]):
#        weekmedian = self.raw.rolling(7,axis=1).median()
#        biweekaverage = weekmedian.rolling(14,axis=1).mean()
#        monthaverage = weekmedian.rolling(28,axis=1).mean()
#        for num in lags:
#            tag = "lag_{0}".format(num)
#            self.df[tag] = self.raw.shift(num,axis=1).stack()
#            self.df["WKM_"+tag] = weekmedian.shift(num,axis=1).stack()
#            self.df["BWKA_"+tag] = biweekaverage.shift(num,axis=1).stack()
#            self.df["MA_"+tag] = monthaverage.shift(num,axis=1).stack()

    def LagFea(self, nums=range(7,63,7),lags=[31,61,91]):
        for num in nums:
            for lag in lags:
                meantag = "mean_{0}_lag_{1}".format(num,lag)
                self.df[meantag] = self.raw.rolling(num,axis=1).mean().shift(lag,axis=1).stack()
                mediantag = "median_{0}_lag_{1}".format(num,lag)
                self.df[mediantag] = self.raw.rolling(num,axis=1).median().shift(lag,axis=1).stack()

    def SeasonLag(self,lags=[31,91]):
        tmp = self.raw.T
        cols = list(tmp.columns.values)
        index = tmp.index
        tmp["date"] = pd.to_datetime(tmp.index)

#        for tag in ["month","day","dayofweek","weekofyear","dayofyear"]:
        for tag in ["dayofweek","day"]:
             if tag == "day":
                 tmp[tag] = tmp.date.dt.day
                 nums = [2,3]
             elif tag == "dayofweek":
                 tmp[tag] = tmp.date.dt.dayofweek
                 nums = [2,4,8,12]
             gp = tmp[cols+[tag]].groupby(tag)
             for num in nums:
                 tmpMean = gp.rolling(num).mean().drop(tag,axis=1).reset_index().sort_values("level_1").set_index("level_1")[cols].T
                 tmpMedian = gp.rolling(num).median().drop(tag,axis=1).reset_index().sort_values("level_1").set_index("level_1")[cols].T
                 for lag in lags:
                     self.df["{0}_mean_{1}_{2}".format(tag,num,lag)] = tmpMean.shift(lag,axis=1).stack()
                     self.df["{0}_median_{1}_{2}".format(tag,num,lag)] = tmpMedian.shift(lag,axis=1).stack()

    def Dummies(self):
        for tag, i, j in zip(["access","agent","lang"],[1,1,0],[0,1,-1]):
            tmp = pd.get_dummies(self.df.index.get_level_values(0)
                                 .map(lambda x: x.split(".wikipedia.org_")[i].split("_")[j]),
                                 prefix=tag)
            for col in list(tmp.columns.values):
                if tag in col:
                    self.df[col] = list(tmp[col])

trainpath = '../../input/train_1.csv'
WT = WebTraffic()
WT.ReadTrain(trainpath)
#WT.WKMediaLag()
WT.LagFea()
WT.SeasonLag()
WT.Dummies()
df = WT.df.dropna(how='any')
#print df

#find feature and target names
features = list(df.columns.values)
target = 'traffic'
features.remove(target)

#
#split train and test
print("split")
train = df[df.index.get_level_values(1)<='2016-10-23']
test = df[df.index.get_level_values(1)>='2016-11-01']
test = test[test.index.get_level_values(1)<='2016-12-01']

train_X, train_y = train[features], train[target]
test_X, test_y = test[features], test[target]

print("model")

from xgboost import XGBRegressor
from smape import XGBsmape

xgbr = XGBRegressor(max_depth=9, 
                    learning_rate=0.05, 
                    n_estimators=1000,
                    silent=True, 
                    objective='reg:linear', 
                    nthread=-1, 
                    subsample=0.8, 
                    colsample_bytree=0.8)

xgbr.fit(train_X,train_y,
         eval_set=[(train_X,train_y),(test_X,test_y)],
         eval_metric=XGBsmape,
         early_stopping_rounds=10,
         verbose=True)

