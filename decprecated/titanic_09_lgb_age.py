#
# Name : titanic_09_lbg_age
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : lgb with age filling
# Score : 0.76076
# 



def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}


def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

%matplotlib inline

# Read data
d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')


# Title
d_train['title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_test['title'] = d_test['Name'].apply(get_title).map(Title_Dictionary)

# Family Size
d_train['FamilySize'] = d_train['SibSp'] + d_train['Parch'] + 1
d_test['FamilySize'] = d_test['SibSp'] + d_test['Parch'] + 1


# Person(Sex)
d_train['Person'] = d_train[['Age','Sex']].apply(get_person,axis=1)
d_test['Person'] = d_test[['Age','Sex']].apply(get_person,axis=1)

dummies_titanic  = pd.get_dummies(d_train['Person'])
dummies_titanic.columns = ['Child','Female','Male']

dummies_test = pd.get_dummies(d_test['Person'])
dummies_test.columns = ['Child','Female','Male']

d_train = d_train.join(dummies_titanic)
d_test = d_test.join(dummies_test)

d_train.drop(['Sex', 'Person'],axis=1,inplace=True)
d_test.drop(['Sex', 'Person'],axis=1,inplace=True)

# Embarked
d_train["Embarked"] = d_train["Embarked"].fillna("S")

dummies_titanic  = pd.get_dummies(d_train['Embarked'])
dummies_titanic.columns = ['E0','E1', 'E2']
d_train = d_train.join(dummies_titanic)

dummies_titanic  = pd.get_dummies(d_test['Embarked'])
dummies_titanic.columns = ['E0','E1', 'E2']
d_test = d_test.join(dummies_titanic)

d_train.drop(['Embarked'],axis=1,inplace=True)
d_test.drop(['Embarked'],axis=1,inplace=True)

# Pclass
dummies_titanic  = pd.get_dummies(d_train['Pclass'])
dummies_titanic.columns = ['Class_1','Class_2', 'Class_3']
d_train = d_train.join(dummies_titanic)

dummies_titanic  = pd.get_dummies(d_test['Pclass'])
dummies_titanic.columns = ['Class_1','Class_2', 'Class_3']
d_test = d_test.join(dummies_titanic)

d_train.drop(['Pclass'],axis=1,inplace=True)
d_test.drop(['Pclass'],axis=1,inplace=True)

# Fare
d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)

# Age
# get average, std, and number of NaN values in titanic_df
average_age_titanic   = d_train["Age"].mean()
std_age_titanic       = d_train["Age"].std()
count_nan_age_titanic = d_train["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = d_test["Age"].mean()
std_age_test       = d_test["Age"].std()
count_nan_age_test = d_test["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

d_train["Age"][np.isnan(d_train["Age"])] = rand_1
d_test["Age"][np.isnan(d_test["Age"])] = rand_2


# Drop
d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)

# Encode
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels = [ 'title']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])

# GBM
split_before_y = d_train["Survived"].values
split_before_x = d_train.drop("Survived",axis=1)

X_train,X_test,y_train,y_test = train_test_split(split_before_x,split_before_y,test_size=0.008,random_state=0)

import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 200,
    'learning_rate': 0.003,
    'num_iterations':1500,
    'feature_fraction': 0.52,
    'bagging_fraction': 0.79,
    'bagging_freq': 7,
    'verbose': 0
}


gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=200)

Y_pred = gbm.predict(d_test.drop("PassengerId",axis=1).copy(), num_iteration=gbm.best_iteration)

for i in range(418):
    if Y_pred[i]>=0.51:
        Y_pred[i]=0
    else:
        Y_pred[i]=1

kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": Y_pred.astype('int64')
    })
kaggle_submission.to_csv("titanic_09_lbg_age.csv", index=False)
