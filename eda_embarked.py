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

print(d_train["Embarked"].value_counts())
print(d_train["Embarked"].isnull().sum())


def extAll(df, val, target):
  return df[(df[target] == val)]
 
def extSurvived(df, val, target):
  return df[((df[target] == val) & (df['Survived'] == 1))]
 
# äKãâÇ≤Ç∆ÇÃê∂ë∂é“ó¶
def calSurvivedRate(df, val, target):
    #print(target + "-" + val + " ÅF " + str(len(extAll(df, val, target))))
    # print(target + "-" + val + " ÅF " + str(len(extSurvived(df, val, target))))
    return str(100 * len(extSurvived(df, val, target)) / len(extAll(df, val, target))) + "%"

for i in ["S", "C", "Q"]:
    print(i, ":", calSurvivedRate(d_train, i, "Embarked"))
    
'''
S : 33.69565217391305%
C : 55.357142857142854%
Q : 38.96103896103896%
'''

sns.barplot(x="Embarked",y="Survived",data=d_train)
