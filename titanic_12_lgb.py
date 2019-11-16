#
# Name : titanic_12_lgb
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : lgb
# Score : 0.7655
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


def fill_age(row):
    condition = (
        (d_train_g_m['Sex'] == row['Sex']) & 
        (d_train_g_m['Pclass'] == row['Pclass']) &
        (d_train_g_m['title'] == row['title']) & 
        (d_train_g_m['FamilySize'] == row['FamilySize'])
    ) 
    return d_train_g_m[condition]['Age'].values[0]

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')
d_train_1 = pd.read_csv('train_1.csv')
d_train = pd.concat([d_train, d_train_1])

d_train['title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_test['title'] = d_test['Name'].apply(get_title).map(Title_Dictionary)

d_train['FamilySize'] = d_train['SibSp'] + d_train['Parch'] + 1
d_test['FamilySize'] = d_test['SibSp'] + d_test['Parch'] + 1

# Person
d_train['Person'] = d_train[['Age','Sex']].apply(get_person,axis=1)
d_test['Person'] = d_test[['Age','Sex']].apply(get_person,axis=1)

dummies_titanic  = pd.get_dummies(d_train['Person'])
dummies_titanic.columns = ['Child','Female','Male']

dummies_test = pd.get_dummies(d_test['Person'])
dummies_test.columns = ['Child','Female','Male']

d_train = pd.concat([d_train, dummies_titanic], axis=1)
d_test = pd.concat([d_test, dummies_test], axis=1)

d_train.drop(['Sex', 'Person'],axis=1,inplace=True)
d_test.drop(['Sex', 'Person'],axis=1,inplace=True)


# Embarked
d_train["Embarked"] = d_train["Embarked"].fillna("S")

dummies_titanic  = pd.get_dummies(d_train['Embarked'])
dummies_titanic.columns = ['E0','E1', 'E2']
d_train = pd.concat([d_train, dummies_titanic], axis=1)

dummies_titanic  = pd.get_dummies(d_test['Embarked'])
dummies_titanic.columns = ['E0','E1', 'E2']
d_test = pd.concat([d_test, dummies_test], axis=1)

d_train.drop(['Embarked'],axis=1,inplace=True)
d_test.drop(['Embarked'],axis=1,inplace=True)

# Pclass
dummies_titanic  = pd.get_dummies(d_train['Pclass'])
dummies_titanic.columns = ['Class_1','Class_2', 'Class_3']
d_train = pd.concat([d_train, dummies_titanic], axis=1)

dummies_titanic  = pd.get_dummies(d_test['Pclass'])
dummies_titanic.columns = ['Class_1','Class_2', 'Class_3']
d_test = pd.concat([d_test, dummies_test], axis=1)

d_train.drop(['Pclass'],axis=1,inplace=True)
d_test.drop(['Pclass'],axis=1,inplace=True)

# Fare
d_train["Fare"] = d_train["Fare"].fillna(35.6271884892086)
d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)

# Age
d_train["Age"] = d_train["Age"].fillna(29.69911764705882)
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)

d_train["Age"] = d_train["Age"].fillna(29.69911764705882)
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)

d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels = ['title']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])
    
y_train = d_train["Survived"]
X_train = d_train.drop("Survived",axis=1)
X_test = d_test.drop(d_test.columns[[0,1]],axis=1).copy()
# X_train,X_test,y_train,y_test = train_test_split(split_before_x,split_before_y,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=1 )
results = []
models = []

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

import lightgbm as lgb

for train_indexes, test_indexes in kf.split(X_train, y_train):

    # print(train_indexes, test_indexes)

    lgb_train = lgb.Dataset(X_train.iloc[train_indexes], y_train.iloc[train_indexes])
    lgb_eval = lgb.Dataset(X_train.iloc[test_indexes], y_train.iloc[test_indexes], reference=lgb_train)
    
    gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=200)
    pred_y = gbm.predict(X_train.iloc[test_indexes],  num_iteration=gbm.best_iteration)
    models.append(gbm)

temp_pred = np.zeros((len(X_test), 5))

for fold_, model in enumerate(models):
    pred_ = model.predict(X_test,num_iteration=model.best_iteration)
    temp_pred[:, fold_] = pred_

Y_pred = (np.mean(temp_pred, axis=1) > 0.5).astype(int)
kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": Y_pred
    })
kaggle_submission.to_csv("titanic_12_lgb.csv", index=False)
