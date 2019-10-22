import numpy as nm
import pandas as pd
import seaborn as sns
%matplotlib inline

d_train = pd.read_csv('train.csv')
d_train['Age_1'] = d_train['Age']

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age')
axis2.set_title('Update Null with Mean')

d_train["Age_1"] = d_train["Age_1"].fillna(29.69911764705882)


d_train['Age'].hist(bins=70, ax=axis1)
d_train['Age_1'].hist(bins=70, ax=axis2)
