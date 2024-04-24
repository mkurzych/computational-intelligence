import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

diabetes = pd.read_csv('diabetes.csv')

scaler = StandardScaler()


def fix_class(item):
    if item == "tested_positive":
        return 1
    else:
        return 0


diabetes["class"] = diabetes["class"].apply(fix_class)

train_set, test_set = train_test_split(diabetes.values, test_size=0.3, random_state=285775)

train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]

scaler.fit(train_inputs)

train_inputs = scaler.transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

results = []

for i in range(1, 10):
    for j in range(1, 10):
        mlp = MLPClassifier(hidden_layer_sizes=(i, j), max_iter=500, activation='relu')
        mlp.fit(train_inputs, train_classes)

        predictions_train = mlp.predict(train_inputs)
        print(accuracy_score(predictions_train, train_classes))
        predictions_test = mlp.predict(test_inputs)
        accuracy = accuracy_score(predictions_test, test_classes)
        print(accuracy)
        matrix = confusion_matrix(predictions_test, test_classes)
        print(matrix)
        results.append(((i, j), accuracy, matrix))

print("\nBest result:")
best_result = max(results, key=lambda x: x[1])
print("Architecture:", best_result[0])
print("Accuracy:", best_result[1])
print("Matrix:", best_result[2])


print("\nWorst result:")
worst_result = min(results, key=lambda x: x[1])
print("Architecture:", worst_result[0])
print("Accuracy:", worst_result[1])
print("Matrix:", worst_result[2])


# relu
# Best result:
# Architecture: (5, 7)
# Accuracy: 0.7878787878787878

# logistic
# Best result:
# Architecture: (3, 3)
# Accuracy: 0.7662337662337663

# identity
# Best result:
# Architecture: (1, 7)
# Accuracy: 0.7748917748917749

# tanh
# Best result:
# Architecture: (6, 4)
# Accuracy: 0.8008658008658008

# to czy gorsze są fp czy fn zależy od zastosowania
# w przypadku chorób chcemy unikać fn, ponieważ to
# oznacza że ktoś jest chory a my tego nie wykryliśmy
# w przypadku np. spamu chcemy unikać fp, ponieważ
# oznacza to że coś co nie jest spamem zostało oznaczone
# jako spam

# dla dobrze wytrenowanych sieci zazwyczaj jest delikatna przewaga
# jednego z błędów, a dla sieci które są słabo wytrenowane
# jest dużo większa różnica między nimi
