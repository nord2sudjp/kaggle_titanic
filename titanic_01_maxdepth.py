#
# Name : titanic_01_maxdepth
# Project : Titanic Machine Learning from Disaster https://www.kaggle.com/c/titanic
# Description : EDA + Decision tree with adjusted MAX_DEPTH=8
# Score : 0.75598
# 


import numpy as nm
import pandas as pd
import seaborn as sns

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

# �t�B�[���h�̔�r - �e�X�g�ɖړI�ϐ��ȊO�ő��݂��Ă��Ȃ����ڂ̓h���b�v���Ă��悢
d_train.head()
d_test.head()

# ���R�[�h���̊m�F
d_train.shape
# (891, 12)

d_test.shape
# (418, 11)

d_train.describe()

# �����l�m�F
sns.heatmap(d_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

d_train.count()

'''
PassengerId    891
Survived       891
Pclass         891
Name           891
Sex            891
Age            714
SibSp          891
Parch          891
Ticket         891
Fare           891
Cabin          204
Embarked       889
dtype: int64
'''

d_train.isnull().sum()

'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
'''

d_test.count()
'''
PassengerId    418
Pclass         418
Name           418
Sex            418
Age            332
SibSp          418
Parch          418
Ticket         418
Fare           417
Cabin           91
Embarked       418
dtype: int64
'''

d_test.isnull().sum()

'''
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
'''

# Age,Embarked,Fare�͕ۊǂ���ACabin�͌����l�������̂ō폜

# �s�v���ڂ̍폜
d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin'], axis=1)

# �m�F
d_train.head()
d_test.head()



# �����l�𖄂߂�
# Embarked -> �J�e�S���J���f�[�^�̂Ȃ̂ōŕp�l�ŕ⊮
pd.value_counts(d_train['Embarked'])
'''
S    644
C    168
Q     77
Name: Embarked, dtype: int64
'''

d_train["Embarked"] = d_train["Embarked"].fillna("S")


# Fare -> �������Ȃ��̂Œ����l���g��
d_test['Fare'].mean()
# 35.6271884892086

d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)



# Age -> �A���f�[�^�Ȃ̂Œ����l
d_train['Age'].mean()
d_train["Age"] = d_train["Age"].fillna(29.69911764705882)


d_test['Age'].mean()
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)


# �J�e�S�����f�[�^�̕ϊ�
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

labels = ['Embarked','Sex']
for label in labels:
    d_train[label]=LE.fit_transform(d_train[label])
    d_test[label]=LE.fit_transform(d_test[label])

# decision tree
y_train = d_train["Survived"].values
x_train = d_train[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp"]].values
x_test = d_test[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp"]].values


# scikit-learn�̃C���|�[�g
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=8) # After Gridsearch with titanic_01_z_MaxDepth
dtree.fit(x_train,y_train)

predictions = dtree.predict(x_test)

kaggle_submission = pd.DataFrame({
        "PassengerId": d_test["PassengerId"],
        "Survived": predictions
    })

#  ���L�� # ���O���ƁA csv�`���̃t�@�C���ɏo��
kaggle_submission.to_csv("titanic_01_maxdepth.csv", index=False)
