import pandas as pd
import numpy as np

import time

trainpath = '../../input/train_1.csv'
keypath = '../../input/key_1.csv'

def read_train(trainpath):
    df = pd.read_csv(trainpath,index_col='Page')
    df = df.fillna(0)
    return df

def key2table(keypath):
    df = pd.read_csv(keypath)
    df["link"] = df.Page.map(lambda x: "_".join(x.split("_")[:-1]))
    df["date"] = df.Page.map(lambda x: x.split("_")[-1])
    df["val"] = np.nan
    df = df.pivot_table(index="link",columns='date',values='val')
    return df

start = time.time()
df = read_train(trainpath)
print time.time()-start
df = df.merge(key2table(keypath),how="left",left_index=True,right_index=True)
print time.time()-start
print df
