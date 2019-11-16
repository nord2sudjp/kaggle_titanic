import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import math
from random import random

%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

d_train["Sex"].value_counts()

all_x = len(d_train)
all_male = len(d_train[d_train["Sex"]=='male'])
all_female = len(d_train[d_train["Sex"]=='female'])
live_male = len(d_train[((d_train['Sex'] == 'male') & (d_train['Survived'] == 1))])
live_female = len(d_train[((d_train['Sex'] == 'female') & (d_train['Survived'] == 1))])
live_all = len(d_train[d_train["Survived"]==1])

print("#### Survive Ratio - Male ####")
print("Male - all: ", all_male)
print("Male - survive: ", live_male)
print("Male - ratio: ", 100 * live_male / all_male, "%")

print("#### Survive Ratio - Female ####")
print("Female - all: ", all_female)
print("Female - survive: ", live_female)
print("Female - ratio: ", 100 * live_female / all_female, "%")

print("#### Survive Ratio - All ####")
print("Female - all: ", all_x)
print("Female - survive: ", live_all)
print("Female - ratio: ", 100 * live_all / all_x, "%")

# visual : Sex, Absolute
d_train["Sex"].value_counts().plot(kind='bar')

# visual : Sex by Survived, Ratio
sns.barplot(x="Sex",y="Survived",data=d_train)

# visual : 
#    Sex by survived, Absolute, Stacked
#    Sex by survived, Ratio, Stacked
d_train['Died'] = 1 - d_train['Survived']
d_train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, colors=['g', 'r']);
d_train.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), 
                                                           stacked=True, colors=['g', 'r']);

