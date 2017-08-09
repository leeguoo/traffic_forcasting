import pandas as pd
import numpy as np


class WebTraffic(object):
    def __init__(self,trainpath):
        flag = 0
        for raw in pd.read_csv(trainpath,index_col='Page',chunksize=100):
            self.raw = raw.applymap(np.log1p)
            #self.raw = pd.read_csv(trainpath,index_col='Page',nrows=1)
            start = self.raw.columns.max()
            end = '2017-03-01'
    
            self.raw = pd.concat([self.raw,
                                  pd.DataFrame(columns=pd.date_range(start,end)[1:]
                                                         .map(lambda t: t.strftime('%Y-%m-%d')))])
            self.raw = self.raw.T.T
    
            self.df = pd.DataFrame()
            self.df["traffic"] = self.raw.stack(dropna=False)#.map(np.log1p)
    
            self.RunAll()

            if flag== 0:
                self.df.to_csv('../../data/data.csv',index=False)
                flag += 1
            else:
                f = open('../../data/data.csv','a')
                self.df.to_csv(f,index=False,header=False)
                f.close()

    def RunAll(self):
        self.WKMediaLag()
        self.LagFea()
        self.SeasonLag()
        self.Agent()
        self.Access()
        self.Lang()
        self.df = self.df[self.df.index.get_level_values(1)>='2016-01-01']
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'level_0':'Page','level_1':'Date'})

    def WKMediaLag(self, lags=[61,91,121,151,181]):
        weekmedian = self.raw.rolling(7,axis=1).median()
        biweekaverage = weekmedian.rolling(14,axis=1).mean()
        monthaverage = weekmedian.rolling(28,axis=1).mean()
        for num in lags:
            tag = "lag_{0}".format(num)
            self.df[tag] = self.raw.shift(num,axis=1).stack(dropna=False)
            self.df["WKM_"+tag] = weekmedian.shift(num,axis=1).stack(dropna=False)
            self.df["BWKA_"+tag] = biweekaverage.shift(num,axis=1).stack(dropna=False)
            self.df["MA_"+tag] = monthaverage.shift(num,axis=1).stack(dropna=False)

    def LagFea(self, nums=range(7,63,7),lags=[61,91,121]):
        for num in nums:
            for lag in lags:
                meantag = "mean_{0}_lag_{1}".format(num,lag)
                self.df[meantag] = self.raw.rolling(num,axis=1).mean().shift(lag,axis=1).stack(dropna=False)
                mediantag = "median_{0}_lag_{1}".format(num,lag)
                self.df[mediantag] = self.raw.rolling(num,axis=1).median().shift(lag,axis=1).stack(dropna=False)

    def SeasonLag(self,lags=[61,91]):
        tmp = self.raw.T
        cols = list(tmp.columns.values)
        index = tmp.index
        tmp["date"] = pd.to_datetime(tmp.index)

        for tag in ["dayofweek","day"]:
             if tag == "day":
                 tmp[tag] = tmp.date.dt.day
                 nums = [1,2,3]
             elif tag == "dayofweek":
                 tmp[tag] = tmp.date.dt.dayofweek
                 nums = [1,2,4,8,12]
             gp = tmp[cols+[tag]].groupby(tag)
             for num in nums:
                 tmpMean = gp.rolling(num).mean().drop(tag,axis=1).reset_index().sort_values("level_1").set_index("level_1")[cols].T
                 tmpMedian = gp.rolling(num).median().drop(tag,axis=1).reset_index().sort_values("level_1").set_index("level_1")[cols].T
                 for lag in lags:
                     self.df["{0}_mean_{1}_{2}".format(tag,num,lag)] = tmpMean.shift(lag,axis=1).stack(dropna=False)
                     self.df["{0}_median_{1}_{2}".format(tag,num,lag)] = tmpMedian.shift(lag,axis=1).stack(dropna=False)

    def Agent(self):
        tmp = pd.get_dummies(self.df.index.get_level_values(0).map(lambda x: x.lower().split("_")[-1]),
                             drop_first=True,
                             prefix="agent")
        for col in list(tmp.columns.values):
            self.df[col] = list(tmp[col])


    def Access(self):
        tmp = pd.get_dummies(self.df.index.get_level_values(0).map(lambda x: x.lower().split("_")[-2]),
                             prefix="access")
        for col in list(tmp.columns.values):
            if "access" in col:
                self.df[col] = list(tmp[col])

    def Lang(self):
        def getLang(x):
            if '.wikipedia.org' in x.lower():
                return x.lower().split(".wikipedia.org")[0].split("_")[-1]
            else:
                return "media"
        tmp = pd.get_dummies(self.df.index.get_level_values(0).map(getLang))
        for col in list(tmp.columns.values):
            self.df[col] = list(tmp[col])

trainpath = '../../input/train_1.csv'
WT = WebTraffic(trainpath)
