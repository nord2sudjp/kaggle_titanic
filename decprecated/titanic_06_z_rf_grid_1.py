#
# Name : titanic_06_z_grid
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : random forest with dummy value
# Score : 
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

Y_train = d_train["Survived"].values
X_train = d_train.drop("Survived",axis=1)
X_train,X_test,y_train,y_test = train_test_split(split_before_x,split_before_y,test_size=0.003,random_state=0)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [3],
    'min_samples_leaf': [3],
    'min_samples_split': [8],
    'n_estimators': [100, 200, 300]
}
grid = GridSearchCV(random_forest, param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)

print(grid.best_params_) 
print(grid.best_score_)
cv_results = grid.cv_results_
x = pd.DataFrame(cv_results)
x.to_csv("titanic_rf_z_grid_1.csv")

'''
{'bootstrap': True, 'max_depth': 100, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 200}
0.8314606741573034
'''
