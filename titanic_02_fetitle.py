#
# Name : titanic_02_fetitle
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : Feature Engineer with Title
# Score : 0.78947
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

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

d_train['title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_test['title'] = d_test['Name'].apply(get_title).map(Title_Dictionary)

d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin'], axis=1)
d_train["Embarked"] = d_train["Embarked"].fillna("S")
d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)
d_train["Age"] = d_train["Age"].fillna(29.69911764705882)
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels = ['Embarked','Sex', 'title']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])
y_train = d_train["Survived"].values
x_train = d_train[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp", "title"]].values
x_test = d_test[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp", "title"]].values

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=8)
dtree.fit(x_train,y_train)

predictions = dtree.predict(x_test)

kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": predictions
    })
    
kaggle_submission.to_csv("titanic_02_fetitle.csv", index=False)
