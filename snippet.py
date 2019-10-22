# �e�L�X�g�f�[�^�̍폜
d_train_w = d_train.fillna({'kcal':0})
d_train_w = d_train_w.drop('event', axis=1)
d_train_w = d_train_w.drop('remarks', axis=1)
d_train_w = d_train_w.drop('precipitation', axis=1)
d_train_w = d_train_w.drop('name', axis=1)
d_train_w = d_train_w.drop('payday', axis=1)

# �����f�[�^�̊���
def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '������', 1 : '%'})
        return kesson_table_ren_columns
 
kesson_table(train)
kesson_table(test)

# �����l
d_train = d_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
d_test = d_test.drop(['Name','Ticket','Cabin'], axis=1)
d_train["Embarked"] = d_train["Embarked"].fillna("S")
d_test["Fare"] = d_test["Fare"].fillna(35.6271884892086)
d_train["Age"] = d_train["Age"].fillna(29.69911764705882)
d_test["Age"] = d_test["Age"].fillna(30.272590361445783)


# Embarked�Ɋ֘A���鏈��
pd.value_counts(train['Embarked'])
df[['Embarked','PassengerId']].groupby('Embarked').count()
train["Embarked"] = train["Embarked"].fillna("S")


# Sex�Ɋ֘A���鏈�� �� ����̓J�e�S�����𐔒l�ϊ����Ă���̂Ŏg��Ȃ�
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


df['Sex'] = df['Sex'].replace('male', 1)
df['Sex'] = df['Sex'].replace('female', 0)

df_final['Sex'] = df_final['Sex'].replace('male', 1)
df_final['Sex'] = df_final['Sex'].replace('female', 0)

# Embarked��Sex�Ɋ֘A���鐳�������� LabelEncoder
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

labels = ['Embarked','Sex']
for label in labels:
    train[label]=LE.fit_transform(train[label])
    test[label]=LE.fit_transform(test[label])


# Feature Engineering for Title
def cvt_title(title):
    if title in ['Rev', 'the Countess', 'Jonkheer']:
        return ''
    elif title in ['Dr', 'Col', 'Major', 'Capt', 'Sir', 'Don']:
        return 'Mr'
    elif title in ['Mlle','Ms', 'Lady']:
        return 'Miss'
    elif title in ['Mme']:
        return 'Mrs'
    else:
        return ''
        
d_train['title'] = d_train['Name'].apply(get_title).apply(cvt_title)
d_test['title'] = d_test['Name'].apply(get_title).apply(cvt_title)

# FE - Familiy Size
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
pd.crosstab(train['FamilySize'], train['Survived']).plot(kind='bar', stacked=True, title="Survived by family size")
pd.crosstab(train['FamilySize'], train['Survived'], normalize='index').plot(kind='bar', stacked=True, title="Survived by family size (%)")

# ����؂̍쐬

my_tree_one = tree.DecisionTreeClassifier()

## �utrain�v�̖ړI�ϐ��Ɛ����ϐ��̒l���擾
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare", "SibSp"]].values

## ���f���\�z & �\��
my_tree_one = my_tree_one.fit(features_one, target)
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values #�utest�v�̐����ϐ��̒l���擾
my_prediction = my_tree_one.predict(test_features) # �utest�v�̐����ϐ����g���āumy_tree_one�v�̃��f���ŗ\��


# ��o-1
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": result
    })

submission.to_csv("submission.csv", index=False)

# ��o-2
PassengerId = np.array(test["PassengerId"]).astype(int) # PassengerId �J�e�S�����f�[�^�^�ϊ�
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"]) # my_prediction(�\���f�[�^�j��PassengerId���f�[�^�t���[���֗��Ƃ�����
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"]) # my_tree_one.csv�Ƃ��ď����o��



# ���̑��Q�l�T�C�g
# https://www.kaggle.com/bonotake/note-on-titanic-tutorial-in-japanese
# https://www.kaggle.com/kojitakahashi6/titanic-koji
# https://www.codexa.net/kaggle-titanic-beginner/