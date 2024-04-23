import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285775)


def order(item):
    return item[-1]


irises_sorted = sorted(train_set, key=order)

for iris in irises_sorted:
    print(iris)


def classify_iris(sl, sw, pl, pw):
    if sl < 6 and pw < 1 and pl < 2 and sw > 2.9:
        return "Setosa"
    elif sl > 4.9 and pw > 1.7 and pl > 4.8 and sw <= 3.8:
        return "Virginica"
    else:
        return "Versicolor"


good_predictions = 0
length = test_set.shape[0]

for i in range(length):
    if classify_iris(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]:
        good_predictions += 1

print(good_predictions)
print(good_predictions / length * 100, "%")
