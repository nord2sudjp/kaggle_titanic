import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

pd.crosstab(d_train['Age'], d_train['Survived']).plot(kind='bar', stacked=True, title="Survived by Age")

d_train=d_train.dropna(subset=['Age'])
figure = plt.figure(figsize=(25, 7))
plt.hist([d_train[d_train['Survived'] == 1]['Age'], d_train[d_train['Survived'] == 0]['Age']], 
         bins=100,stacked=True, color = ['g','r'],
         label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend();


d_train = pd.read_csv('train.csv')
d_train['title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_train['FamilySize'] = d_train['SibSp'] + d_train['Parch'] + 1
d_train_g = d_train_g = d_train.groupby(['Sex','Pclass','title', 'FamilySize'])
d_train_g_m = d_train_g.median()
d_train_g_m.reset_index()[['Sex', 'Pclass', 'title', 'FamilySize', 'Age']]
print(d_train_g_m)
