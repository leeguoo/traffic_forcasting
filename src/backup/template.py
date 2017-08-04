import pandas as pd

def train_stack(df):
    df = df.stack().reset_index()
    df["Page"] = df.Page+"_"+df.level_1
    df.drop(["level_1"],axis=1,inplace=True)
    df = df.rename(columns={0:"traffic"})
    return df

def parse_link(df):
    df["link"] = df["Page"].map(lambda x: "_".join(x.split("_")[:-1]))
    return df

def parse_date(df):
    df["date"] = pd.to_datetime(df["Page"].map(lambda x: x.split("_")[-1]))
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["day"] = df.date.dt.day
    df["weekday"] = df.date.dt.dayofweek
    return df

df = pd.read_csv('../input/train_1.csv',index_col='Page',nrows=3)

df = train_stack(df)
df = parse_link(df)
df = parse_date(df)
#df = df.stack().reset_index()
#
#df["date"] = pd.to_datetime(df.level_1)
#df["year"] = df.date.dt.year
#df["month"] = df.date.dt.month
#df["day"] = df.date.dt.day
#df["weekday"] = df.date.dt.dayofweek
#
#df["lang"] = df.Page.map(lambda x: x.split(".wikipedia.org_")[0].split("_")[-1])
#df["title"] = df.Page.map(lambda x: x.split(".wikipedia.org_")[0].split("_")[0])
#df["access"] = df.Page.map(lambda x: x.split(".wikipedia.org_")[1].split("_")[0])
#df["agent"] = df.Page.map(lambda x: x.split(".wikipedia.org_")[1].split("_")[1])
#
#df["Page"] = df.Page+"_"+df.level_1
#df.drop(["level_1"],axis=1,inplace=True)
#df = df.rename(columns={0:"traffic"})

print(df)
