import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285775)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)

r = export_text(dtc, feature_names=df.keys()[:-1])
print(r)

train_pred = dtc.predict(train_inputs)
test_pred = dtc.predict(test_inputs)

print("Accuracy for the training set")
print(dtc.score(train_inputs, train_classes))
# 1.0

print("Accuracy for the test set")
print(dtc.score(test_inputs, test_classes))
# 0.9(3)

cm_train = confusion_matrix(train_classes, train_pred)
cm_test = confusion_matrix(test_classes, test_pred)

print("Confusion matrix for the training set")
print(cm_train)

print("Confusion matrix for the test set")
print(cm_test)


