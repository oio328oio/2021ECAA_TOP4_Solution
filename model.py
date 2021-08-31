#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import string
import random
import time
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
# lightgbm: '3.2.1'
# pandas:'1.2.3'
# numpy: '1.19.2'
# python: '3.7.3'


# In[ ]:


class Timer:
    def __init__(self):
        self.s_time = time.time()

    def start(self):
        self.s_time = time.time()

    def now(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def elasped_time(self):
        ela_time = time.time() - self.s_time
        self.s_time = time.time()
        if ela_time < 60:
            return "%.2f s"%ela_time
        
        if ela_time < 3600:
            minute =  ela_time // 60
            second = ela_time % 60
            return "%i:%.2f s"%(minute,second)
        hour = ela_time // 3600
        minute =  (ela_time - hour*3600) // 60
        second = (ela_time - hour*3600) % 60
        return "%i:%i:%.2f s"%(hour,minute,second)
        
        
        
timer = Timer() 
def print_log(*args,start=False):
    if start:
        print("Start",*args,"at", timer.now())
    else:
        print("Finish",*args,"at %s,"%timer.now()," elasped time: ",timer.elasped_time())
    
print_log(start=True)


# In[ ]:


train = pd.read_csv("./data/train.csv")
target = train.pop("orders_3h_15h")

test = pd.read_csv("./data/test.csv")
article_id = test["article_id"].values[:]
orders_2h = test["orders_2h"]
train_num = len(train)
data = pd.concat([train,test])


# In[ ]:


data["day_of_week"] = (data["date"])%7
data["day_of_month"] = (data["date"])%30


def reset_article(data,cols):
    artcile_ids = data.groupby(cols)["article_id"].unique()
    artcile_index_map = dict()
    for index in artcile_ids.index:
        artcile_index_map.update({a:i for i,a in enumerate(np.sort(artcile_ids[index]))})
    
    col = cols if isinstance(cols,str) else cols[1]
    data['reset_article_id_by_%s'%col] = data['article_id'].map(artcile_index_map)
    print_log("reset_article",cols)
    return data



# In[ ]:


data = reset_article(data,"date")
data = reset_article(data,"baike_id_2h")
data = reset_article(data,"baike_id_1h")


# In[ ]:


for col in data.columns:
    if data[col].dtype == "int64":
        data[col] = data[col].astype("int32")
    elif data[col].dtype == "float64":
        data[col] = data[col].astype("float32")


# In[ ]:



data['comments'] = data['comments_2h'] - data["comments_1h"]
data['buzhi'] = data['buzhi_2h'] - data["buzhi_1h"]
data['favorite'] = data['favorite_2h'] - data["favorite_1h"]

data['orders'] = data['orders_2h'] - data["orders_1h"]
data["positive_1h"] = data["favorite_1h"] + data["comments_1h"] + data["zhi_1h"] + data["orders_1h"]
data["positive_2h"] = data["favorite_2h"] + data["comments_2h"] + data["zhi_2h"] + data["orders_2h"]
data["positive"] = data["positive_2h"] - data["positive_1h"]
data['orders_aid'] = data['orders_2h'] * (data['reset_article_id_by_date'] + 1)


# In[ ]:


for col in data.columns:
    if "article_id" == col:
        continue
    cnt = data.groupby(col)["article_id"].count().reset_index()
    name = "%s_div"%col
    cnt.rename(columns={"article_id":name},inplace=True)
    cnt[name] = np.log(1+1/(cnt[name]+1))
    data = data.merge(cnt,on=col,how="left") 

    if 'orders_2h' == col:
        continue
    
    cnt = data.groupby(col)["orders_2h"].mean().reset_index()
    name = "%s_mean_div"%col
    cnt.rename(columns={"orders_2h":name},inplace=True)
    cnt[name] = np.log(1+1/(cnt[name]+1))
    data = data.merge(cnt,on=col,how="left") 
    print_log("div",col)


# In[ ]:


def rename(prefix,suffix,group_cols):
    def _rename(col):
        if col in group_cols:
            return col
        _suffix = "a"
        return ("group_%s_%s_%s_%s"%(prefix,suffix,col,_suffix)).replace("'","")
    return _rename


# In[ ]:


def feature_stat(data:pd.DataFrame,group_cols,agg_cols,method):
    np_agg = stat_methods[method]
    stat_feature = data.groupby(group_cols)[agg_cols].agg(np_agg).reset_index()
    stat_feature.rename(columns=rename(tuple(group_cols),method,group_col),inplace=True)
    return stat_feature

def feature_count(data:pd.DataFrame,group_cols,agg_cols):
    count_feature = data.groupby(group_cols)[agg_cols].count().reset_index()
    col = count_feature.columns[-1]
    count_feature[col] = count_feature[col].astype("int")
    count_feature.rename(columns=rename(tuple(group_cols),"count",group_col),inplace=True)
    return count_feature


# In[ ]:


agg_cols = ["orders_2h","orders_1h","favorite_2h","price","favorite_1h","price_diff","comments_2h","zhi_2h",
           "comments_1h","buzhi_2h","buzhi_1h"]

stat = data.groupby(["mall","date"])[agg_cols].mean().reset_index()
stat.rename(columns=rename("g","m",("mall","date")),inplace=True)
data = data.merge(stat,on=["mall","date"],how="left")
print_log("stat",["mall","date"])
    
stat = data.groupby(["baike_id_1h","url"])[agg_cols].mean().reset_index()
stat.rename(columns=rename("h","m",("baike_id_1h","url")),inplace=True)
data = data.merge(stat,on=["baike_id_1h","url"],how="left")
for col in agg_cols:
    data["%s_minus_mean_h"%col] = data[col] - data["group_h_m_%s_a"%col]
print_log("stat",["baike_id_1h","url"])
    
    
stat = data.groupby(["mall","author"])[agg_cols].mean().reset_index()
stat.rename(columns=rename("c","m",("mall","author")),inplace=True)
data = data.merge(stat,on=["mall","author"],how="left")
print_log("stat",["mall","author"])
    
stat = data.groupby(["url","author"])[agg_cols].mean().reset_index()
stat.rename(columns=rename("k","m",("url","author")),inplace=True)
data = data.merge(stat,on=["url","author"],how="left")
print_log("stat",["url","author"])


# In[ ]:


for col in data.columns:
    if data[col].dtype == "int64":
        data[col] = data[col].astype("int32")
    elif data[col].dtype == "float64":
        data[col] = data[col].astype("float32")


# In[ ]:


data = np.array(data)
train = data[:train_num]
test = data[train_num:]
target = np.array(target)


# In[ ]:


kfold = StratifiedKFold(n_splits=5,random_state=2021,shuffle=True)


# In[ ]:


preds = 0
fold_num = 5
for idx,(train_index,valid_index) in enumerate(kfold.split(train,target)):
    print(idx,'split data ......')
    x_train = train[train_index]
    y_train = target[train_index]

    x_valid = train[valid_index]
    y_valid = target[valid_index]

    model = lgb.LGBMRegressor(    
                        objective='poisson',
                        metric='mse',
                        boosting='gbdt',
                        num_leaves= 2 ** 8,
                        bagging_fraction=0.8,
                        bagging_freq= 10,
                        bagging_seed= 17,
                        feature_fraction=0.8,
                        feature_fraction_seed= 17,
                        max_depth = -1,
                        learning_rate= 0.01, 
                        n_estimators= 6000,
                        max_bin = 400,
                        lambda_l1= 100,
                        lambda_l2= 100,
                        importance_type='gain'
                        )
    print(idx,'train ......')
    model.fit(
              x_train,y_train,
              eval_set = [(x_train,y_train),(x_valid,y_valid)],
              eval_metric="mse",
              early_stopping_rounds=200,
              verbose=True,
             )

    print(idx, 'eval ......')
    pred_y = model.predict(test)
    preds = preds + pred_y
    del x_train,x_valid
    gc.collect()
    


# In[ ]:


preds_ = preds/fold_num
df = pd.DataFrame()
df["article_id"] = article_id
df["orders_3h_15h"] = preds_
df.to_csv("./sample_submission.csv",index=False)

