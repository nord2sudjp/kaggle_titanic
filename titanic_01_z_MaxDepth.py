# Search best MAX_DEPTH for Decision Tree

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


MAX_DEPTH = 20
depths = range(1, MAX_DEPTH)

loo_y = d_train["Survived"].values
loo_X = d_train[["Pclass", "Sex", "Age", "Fare", "Parch", "Embarked", "SibSp"]].values

accuracy_scores = []
for depth in depths:

    predicted_labels = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(loo_X):
            X_train, X_test = loo_X[train_index], loo_X[test_index]
            y_train, y_test = loo_y[train_index], loo_y[test_index]


            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_train, y_train)

            predicted_label = clf.predict(loo_X[test_index])
            predicted_labels.append(predicted_label)

    score = accuracy_score(loo_Y, predicted_labels)
    print('max depth={0}: {1}'.format(depth, score))

    accuracy_scores.append(score)

# Å‘å[“x‚²‚Æ‚Ì³‰ğ—¦‚ğÜ‚êüƒOƒ‰ƒt‚Å‰Â‹‰»‚·‚é
X = list(depths)
plt.plot(X, accuracy_scores)
plt.xlabel('max depth')
plt.ylabel('accuracy rate')
plt.show()
