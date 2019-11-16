# 0.80382
import pandas as pd
sub_02 = pd.read_csv("./ensamble/titanic_02_fetitle.csv")
sub_03 = pd.read_csv("./ensamble/titanic_03_familysize.csv")
sub_05 = pd.read_csv("./ensamble/titanic_05_rf.csv")
sub_081 = pd.read_csv("./ensamble/titanic_08_lgb_1.csv")
sub_082 = pd.read_csv("./ensamble/titanic_08_lgb_2.csv")

sub = pd.DataFrame(pd.read_csv("./test.csv")['PassengerId'])

sub['Survived'] = sub_02['Survived'] + sub_03['Survived'] + sub_05['Survived'] + sub_081['Survived'] + sub_082['Survived']
sub['Survived'] = (sub['Survived'] >= 3).astype(int)
sub.to_csv("ensemble.csv", index = False)
