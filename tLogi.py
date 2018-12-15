# encoding: utf-8
# ロジスティック回帰、
# 標準化を使用すると。各説明変数の　数単位の影響が少なくする事ができる。
#

# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
#%matplotlib inline
# 機械学習モジュール
import sklearn

# 標準化対応、学習。
# 学習データ
adult_data = pd.read_csv("dat_money.csv" )
print(adult_data.head( ))

#
adult_data.info()
#
adult_data.groupby("flg-50K").size()
#
# 目的変数：flg立てをする
adult_data["fin_flg"] = adult_data["flg-50K"].map(lambda x: 1 if x ==' >50K' else 0)

#
adult_data.groupby("fin_flg").size()

#
# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 標準化のためのモジュール
from sklearn.preprocessing import StandardScaler

# 説明変数と目的変数
X = adult_data[["age","fnlwgt","education-num","capital-gain","capital-loss"]]
Y = adult_data['fin_flg']

# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0)

# ロジスティック回帰
model = LogisticRegression()

# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

clf = model.fit(X_train_std,y_train)
print("train:",clf.score(X_train_std,y_train))
print("test:", clf.score(X_test_std,y_test))

print(clf.coef_ )
#
pred= model.predict(X_test_std[:10])
print(pred )
quit()
#
# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 説明変数と目的変数
X = adult_data[["age","fnlwgt","education-num","capital-gain","capital-loss"]]
Y = adult_data['fin_flg']

# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0)

# ロジスティック回帰のインスタンス
model = LogisticRegression()

# モデルのあてはめ
clf = model.fit(X_train,y_train)

print("train result:",clf.score(X_train,y_train))
print("test result:" , clf.score(X_test,y_test))

