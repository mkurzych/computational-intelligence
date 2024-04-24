from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

iris = load_iris()
scaler = StandardScaler()

datasets = train_test_split(iris.data, iris.target, test_size=0.3)

train_data, test_data, train_labels, test_labels = datasets

scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

neurons = [2, 3, (3, 3)]
iterations = [1000, 2000, 3000]
results = []
for iteration in iterations:
    print("\nIterations:", iteration)
    for neuron in neurons:
        print("\nArchitecture:", neuron)
        mlp = MLPClassifier(hidden_layer_sizes=neuron, max_iter=iteration)
        mlp.fit(train_data, train_labels)

        predictions_train = mlp.predict(train_data)
        print(accuracy_score(predictions_train, train_labels))
        predictions_test = mlp.predict(test_data)
        print(accuracy_score(predictions_test, test_labels))
        print(confusion_matrix(predictions_test, test_labels))
        results.append((neuron, iteration, accuracy_score(predictions_test, test_labels)))


print("\nBest result:")
best_result = max(results, key=lambda x: x[2])
print("Architecture:", best_result[0])
print("Iterations:", best_result[1])
print("Accuracy:", best_result[2])
