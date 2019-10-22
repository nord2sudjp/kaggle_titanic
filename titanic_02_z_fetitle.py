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

pd.value_counts(d_train['title'])

'''
Mr              517
Miss            182
Mrs             125
Master           40
Dr                7
Rev               6
Major             2
Col               2
Mlle              2
Ms                1
the Countess      1
Capt              1
Jonkheer          1
Lady              1
Sir               1
Don               1
Mme               1
'''
