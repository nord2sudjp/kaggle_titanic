import numpy as nm
import pandas as pd
import seaborn as sns
%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

d_train['FamilySize'] = d_train['SibSp'] + d_train['Parch'] + 1
d_test['FamilySize'] = d_test['SibSp'] + d_test['Parch'] + 1

pd.crosstab(d_train['FamilySize'], d_train['Survived']).plot(kind='bar', stacked=True, title="Survived by family size")
pd.crosstab(train['FamilySize'], d_train['Survived'], normalize='index').plot(kind='bar', stacked=True, title="Survived by family size (%)")
