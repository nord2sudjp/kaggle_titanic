#
# Name : titanic_01_maxdepth
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : EDA + Decision tree with adjusted MAX_DEPTH=8
# Score : 0.75598
# 


import numpy as nm
import pandas as pd
import seaborn as sns

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

# フィールドの比較 - テストに目的変数以外で存在していない項目はドロップしてもよい
d_train.head()
d_test.head()

# レコード数の確認
d_train.shape
# (891, 12)

d_test.shape
# (418, 11)

d_train.describe()

# 欠損値確認
sns.heatmap(d_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

d_train.count()

'''
PassengerId    891
Survived       891
Pclass         891
Name           891
Sex            891
Age            714
SibSp          891
Parch          891
Ticket         891
Fare           891
Cabin          204
Embarked       889
dtype: int64
'''

d_train.isnull().sum()

'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
'''

d_test.count()
'''
PassengerId    418
Pclass         418
Name           418
Sex            418
Age            332
SibSp          418
Parch          418
Ticket         418
Fare           417
Cabin           91
Embarked       418
dtype: int64
'''

d_test.isnull().sum()

'''
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
'''

# Age,Embarked,Fareは保管する、Cabinは欠損値が多いので削除

# 不要項目の削除
d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin'], axis=1)

# 確認
d_train.head()
d_test.head()



# 欠損値を埋める
# Embarked -> カテゴリカルデータのなので最頻値で補完
pd.value_counts(d_train['Embarked'])
'''
S    644
C    168
Q     77
Name: Embarked, dtype: int64
'''

d_train["Embarked"] = d_train["Embarked"].fillna("S")


# Fare -> 件数少ないので中央値を使う
d_test['Fare'].mean()
# 35.6271884892086

d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)



# Age -> 連続データなので中央値
d_train['Age'].mean()
d_train["Age"] = d_train["Age"].fillna(29.69911764705882)


d_test['Age'].mean()
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)


# カテゴラルデータの変換
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

labels = ['Embarked','Sex']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])

# decision tree
y_train = d_train["Survived"].values
x_train = d_train[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp"]].values
x_test = d_test[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp"]].values


# scikit-learnのインポート
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=8) # After Gridsearch with titanic_01_z_MaxDepth
dtree.fit(x_train,y_train)

predictions = dtree.predict(x_test)

kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": predictions
    })

#  下記の # を外すと、 csv形式のファイルに出力
kaggle_submission.to_csv("titanic_01_maxdepth.csv", index=False)
