import pandas as pd

#read raw data

class WebTraffic(object):
    def __init__(self):
        pass

    def ReadTrain(self,trainpath):
        self.raw = pd.read_csv(trainpath,index_col='Page',nrows=1000)
        self.raw = self.raw.fillna(0)

        self.df = pd.DataFrame()
        self.df["traffic"] = self.raw.stack()

    def WKMediaLag(self, lags=[61,91,121,151,181]):
        weekmedian = self.raw.rolling(7,axis=1).median()
        biweekaverage = weekmedian.rolling(14,axis=1).mean()
        monthaverage = weekmedian.rolling(28,axis=1).mean()
        for num in lags:
            tag = "lag_{0}".format(num)
            self.df[tag] = self.raw.shift(num,axis=1).stack()
            self.df["WKM_"+tag] = weekmedian.shift(num,axis=1).stack()
            self.df["BWKA_"+tag] = biweekaverage.shift(num,axis=1).stack()
            self.df["MA_"+tag] = monthaverage.shift(num,axis=1).stack()

    def Season(self):
        tmp = self.raw.T
        cols = list(tmp.columns.values)
        index = tmp.index
        tmp["date"] = pd.to_datetime(tmp.index)

#        for tag in ["month","day","dayofweek","weekofyear","dayofyear"]:
        for tag in ["month","day","dayofweek"]:
             if tag == "month":
                 tmp[tag] = tmp.date.dt.month
             elif tag == "day":
                 tmp[tag] = tmp.date.dt.day
             elif tag == "dayofweek":
                 tmp[tag] = tmp.date.dt.dayofweek
             elif tag == "weekofyear":
                 tmp[tag] = tmp.date.dt.weekofyear
             elif tag == "dayofyear":
                 tmp[tag] = tmp.date.dt.dayofyear
             gp = tmp.groupby(tag,as_index=False)
             self.df[tag+"_mean"] = tmp[["date",tag]].merge(gp.mean(),how="left",on=tag).set_index(index)[cols].T.stack()
             self.df[tag+"_median"] = tmp[["date",tag]].merge(gp.median(),how="left",on=tag).set_index(index)[cols].T.stack()

    def Dummies(self):
        #df["title"] = df.Page.map(lambda x: x.split(".wikipedia.org_")[0].split("_")[0])
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
WT.WKMediaLag()
#WT.Season()
WT.Dummies()
df = WT.df.dropna(how='any')#.head(20)
#print df
#
#find feature and target names
features = list(df.columns.values)
target = 'traffic'
features.remove(target)

#
#split train and test
print("split")
train = df[df.index.get_level_values(1)<'2016-11-01']
test = df[df.index.get_level_values(1)>='2016-11-01']

train_X, train_y = train[features], train[target]
test_X, test_y = test[features], test[target]

print("model")
#random forest model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(50,n_jobs=-1)
rf.fit(train_X, train_y)
preds = rf.predict(test_X)


from smape import smape
print(smape(train_y,rf.predict(train_X)))
print(smape(test_y,preds))
