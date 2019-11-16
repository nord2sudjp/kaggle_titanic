#
# Name : titanic_13_rf_02
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : random forest best score
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

def cabin_select():
    if 0.335 > random():
        return 'C'
    elif 0.267 > random():
        return 'B'
    elif 0.153 > random():
        return 'D'
    elif 0.142 > random():
        return 'E'
    else:
        return 'A'

def cabin_select_low():
    if 0.464 > random():
        return 'F'
    elif 0.25 > random():
        return 'E'
    elif 14.3 > random():
        return 'D'
    else:
        return 'G'

import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import random

%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')
#d_train_1 = pd.read_csv('train_1.csv')
#d_train = pd.concat([d_train, d_train_1])

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


# Fare
d_train["Fare"] = d_train["Fare"].fillna(d_train.Fare.mean())
d_test["Fare"] = d_test["Fare"].fillna(d_test.Fare.mean())

# Age
d_train["Age"] = d_train["Age"].fillna(d_train.Age.mean())
d_test["Age"] = d_test["Age"].fillna(d_test.Age.mean())

# Cabin
cabin = d_train['Cabin'].replace('[0-9]+', '', regex=True).replace(r'\s+', r'', regex=True).replace(r'([\w])[\w]+', r'\1', regex=True)
d_train['Cabin_replaced'] = cabin

cabin = d_test['Cabin'].replace('[0-9]+', '', regex=True).replace(r'\s+', r'', regex=True).replace(r'([\w])[\w]+', r'\1', regex=True)
d_test['Cabin_replaced'] = cabin

d_train.loc[(d_train['Pclass'] == 1) & (d_train['Cabin_replaced'].isnull()), "Cabin_replaced"] = cabin_select()
d_train.loc[((d_train['Pclass'] == 2) | (d_train['Pclass'] == 3)) & (d_train['Cabin_replaced'].isnull()), 'Cabin_replaced'] = cabin_select_low()
d_test.loc[(d_test['Pclass'] == 1) & (d_test['Cabin_replaced'].isnull()), "Cabin_replaced"] = cabin_select()
d_test.loc[((d_test['Pclass'] == 2) | (d_test['Pclass'] == 3)) & (d_test['Cabin_replaced'].isnull()), 'Cabin_replaced'] = cabin_select_low()
d_train.drop(['Cabin', 'Pclass'],axis=1,inplace=True)
d_test.drop(['Cabin', 'Pclass'],axis=1,inplace=True)


# Drop
d_train = d_train.drop(['PassengerId','Name','Ticket', 'Parch','SibSp'], axis=1)
d_test = d_test.drop(['Name','Ticket', 'Parch','SibSp'], axis=1)


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels = ['title', 'Cabin_replaced']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])
    
split_before_y = d_train["Survived"]
split_before_x = d_train.drop("Survived",axis=1)
X_train,X_test,y_train,y_test = train_test_split(split_before_x,split_before_y,test_size=0.1,random_state=0)

# {'bootstrap': True, 'max_depth': 80, 'max_features': 2, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 200}
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(bootstrap=True, max_depth=80, max_features=2, min_samples_leaf=3, min_samples_split=10, n_estimators=200)
random_forest.fit(X_train, y_train)

print("Train Score:", random_forest.score(X_train, y_train))
print("Test Score:" , random_forest.score(X_test, y_test))

X_test = d_test.drop(d_test.columns[[0,1]],axis=1).copy()
Y_pred = random_forest.predict(X_test)
        
kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": Y_pred
    })
kaggle_submission.to_csv("titanic_13_rf_02.csv", index=False)
