import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285775)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

k_range = [3, 5, 11]
for k in k_range:
    print("\nKNN Classifier -", k, "neighbors")
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_inputs, train_classes)

    test_pred = classifier.predict(test_inputs)

    accuracy = classifier.score(test_inputs, test_classes)
    print("Accuracy for the test set:", accuracy)

    cm_test = confusion_matrix(test_classes, test_pred)
    print("Confusion matrix for the test set:\n", cm_test)

print("\nNB Classifier")
model = GaussianNB()
model.fit(train_inputs, train_classes)
test_pred = model.predict(test_inputs)
accuracy = model.score(test_inputs, test_classes)
print("Accuracy for the test set:", accuracy)
cm_test = confusion_matrix(test_classes, test_pred)
print("Confusion matrix for the test set:\n", cm_test)

# Podium dla klasyfikatorów:
# 1. 11KNN - (1.0)
# 2. 3KNN, 5KNN, NB - (0.9(7))
# 3. DD - (0.9(3))
