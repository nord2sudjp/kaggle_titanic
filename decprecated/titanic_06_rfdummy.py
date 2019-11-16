#
# Name : titanic_06_rfdummy
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : random forest with dummy value
# Score : 0.74641
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



import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

d_train['title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_test['title'] = d_test['Name'].apply(get_title).map(Title_Dictionary)

d_train['FamilySize'] = d_train['SibSp'] + d_train['Parch'] + 1
d_test['FamilySize'] = d_test['SibSp'] + d_test['Parch'] + 1

person_dummies_titanic = pd.get_dummies(d_train['Sex'])
person_dummies_titanic.columns = ['Female','Male']
d_train = d_train.join(person_dummies_titanic)
d_train.drop(['Sex'], axis=1, inplace=True)

person_dummies_titanic = pd.get_dummies(d_test['Sex'])
person_dummies_titanic.columns = ['Female','Male']
d_test = d_test.join(person_dummies_titanic)
d_test.drop(['Sex'], axis=1, inplace=True)

d_train["Embarked"] = d_train["Embarked"].fillna("S")

d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)

d_train["Age"] = d_train["Age"].fillna(29.69911764705882)
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)

d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels = ['Embarked', 'title']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])


Y_train = d_train["Survived"].values
X_train = d_train.drop("Survived",axis=1)
X_test = d_test.drop(d_test.columns[[0,1]],axis=1).copy()

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

Y_pred = random_forest.predict(X_test)
        
kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": Y_pred
    })
kaggle_submission.to_csv("titanic_06_rfdummy.csv", index=False)
