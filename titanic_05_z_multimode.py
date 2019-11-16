# https://www.kaggle.com/omarelgabry/a-journey-through-titanic?scriptVersionId=447794

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

d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin', 'Parch','SibSp'], axis=1)
d_train["Embarked"] = d_train["Embarked"].fillna("S")
d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)

d_train_g = d_train.groupby(['Sex','Pclass','title', 'FamilySize'])
d_train_g_m = d_train_g.median()
d_train_g_m = d_train_g_m.reset_index()[['Sex', 'Pclass', 'title', 'FamilySize', 'Age']]

# d_train['Age'] = d_train.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
# d_test['Age'] = d_train.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

d_train["Age"] = d_train["Age"].fillna(29.69911764705882)
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels = ['Embarked','Sex', 'title']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])


Y_train = d_train["Survived"].values
X_train = d_train.drop("Survived",axis=1)
X_test = d_test.drop(d_test.columns[[0,1]],axis=1).copy()

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


dtree = DecisionTreeClassifier(max_depth=8)
dtree.fit(X_train, Y_train)
Y_pred = dtree.predict(X_test)
print("dtree", dtree.score(X_train, Y_train))

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print("log", logreg.score(X_train, Y_train))

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print("svc", svc.score(X_train, Y_train))

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print("rf", random_forest.score(X_train, Y_train))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print("knn", knn.score(X_train, Y_train))

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
print("NB", gaussian.score(X_train, Y_train))

'''
dtree 0.8900112233445566
log 0.8024691358024691
svc 0.9046015712682379
rf 0.9842873176206509
knn 0.8406285072951739
NB 0.8080808080808081
'''
