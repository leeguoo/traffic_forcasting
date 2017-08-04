import pandas as pd

#read raw data
trainpath = '../../input/train_1.csv'
df = pd.read_csv(trainpath,nrows=2,index_col='Page').T
df["w"] =  pd.to_datetime(df.index).dayofweek
#df["cs"] = df.iloc[:,0].cumsum()
#df["cs"] = df.groupby("w").cumsum()
print df.groupby("w").rolling(2).mean().drop("w",axis=1).reset_index().sort_values("level_1")

#print df
