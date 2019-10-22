import numpy as np

for name, importance in zip(["Pclass", "Sex", "Age", "Fare",  "Embarked", "FamilySize", "title"], dtree.feature_importances_):
    print(name, importance)

def fill_age(row):
    condition = (
        (d_train_g_m['Sex'] == row['Sex']) & 
        (d_train_g_m['Pclass'] == row['Pclass']) &
        (d_train_g_m['title'] == row['title']) & 
        (d_train_g_m['FamilySize'] == row['FamilySize'])
    ) 
    return d_train_g_m[condition]['Age'].values[0]

d_train = pd.read_csv('train.csv')
d_train['title'] = d_train['Name'].apply(get_title).map(Title_Dictionary)
d_train['FamilySize'] = d_train['SibSp'] + d_train['Parch'] + 1
d_train_g = d_train_g = d_train.groupby(['Sex','Pclass','title', 'FamilySize'])
d_train_g_m = d_train_g.median()
d_train_g_m = d_train_g_m.reset_index()[['Sex', 'Pclass', 'title', 'FamilySize', 'Age']]

d_train['Age'] = d_train.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

d_train['Age'].hist(bins=70)

