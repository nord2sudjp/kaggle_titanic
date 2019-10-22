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


# Embarked
pd.value_counts(train['Embarked'])
df[['Embarked','PassengerId']].groupby('Embarked').count()
train["Embarked"] = train["Embarked"].fillna("S")

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


# �J�e�S�����ϊ�
df['Sex'] = df['Sex'].replace('male', 1)
df['Sex'] = df['Sex'].replace('female', 0)

df_final['Sex'] = df_final['Sex'].replace('male', 1)
df_final['Sex'] = df_final['Sex'].replace('female', 0)


from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

labels = ['Embarked','Sex']
for label in labels:
    train[label]=LE.fit_transform(train[label])
    test[label]=LE.fit_transform(test[label])
    

# ����؂̍쐬
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
# �utest�v�̐����ϐ��̒l���擾
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# �utest�v�̐����ϐ����g���āumy_tree_one�v�̃��f���ŗ\��
my_prediction = my_tree_one.predict(test_features)

# �utrain�v�̖ړI�ϐ��Ɛ����ϐ��̒l���擾
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare", "SibSp"]].values



# ��o
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": result
    })

submission.to_csv("submission.csv", index=False)


# 
PassengerId = np.array(test["PassengerId"]).astype(int)
 
# my_prediction(�\���f�[�^�j��PassengerId���f�[�^�t���[���֗��Ƃ�����
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
 
# my_tree_one.csv�Ƃ��ď����o��
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])

# ���̑��Q�l�T�C�g
# https://www.kaggle.com/bonotake/note-on-titanic-tutorial-in-japanese
# https://www.kaggle.com/kojitakahashi6/titanic-koji
# https://www.codexa.net/kaggle-titanic-beginner/