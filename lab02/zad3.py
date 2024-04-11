# Stwórz wykresy z irysami jako punktami na wykresie, dla
# dwóch zmiennych: sepal length i sepal width. Klasy irysów
# oznaczone są w legendzie wykresu. Zrób wykres w trzech
# wersjach: dane oryginalne, znormalizowane min-max i zeskalowane
# z-scorem. Wynik powinien przypominać ten poniżej. Co możesz
# powiedzieć o min, max, mean, standard deviation dla tych danych?

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

irises = pd.read_csv("iris1.csv")

versions = [('Original', None), ('Min-Max Normalized', MinMaxScaler()), ('Z-Score Normalized', StandardScaler())]
iris_varieties = ["Setosa", "Versicolor", "Virginica"]

fig, axs = plt.subplots(len(versions), figsize=(10, 15))

i = 0
for version in versions:
    X = irises[["sepal.length", "sepal.width"]].values
    y = irises["variety"].values
    if version[1]:
        X = version[1].fit_transform(X)
    for name in iris_varieties:
        axs[i].scatter(X[y == name, 0], X[y == name, 1], label=name)
    axs[i].set_xlabel("sepal.length")
    axs[i].set_ylabel("sepal.width")
    axs[i].set_title(version[0])
    axs[i].legend()
    i += 1

    # print("Min:", X.min())
    # print("Max:", X.max())
    # print("Mean:", X.mean())
    # print("Standard deviation:", X.std())

plt.show()

# Wnioski:
# Min, max, mean, standard deviation dla danych oryginalnych, znormalizowanych min-max
# i zeskalowanych z-scorem są różne.
# Dla danych oryginalnych min = 2, max = 7.9, mean = 4.45, standard deviation = 1.54.
# Dla danych znormalizowanych min-max min = 0, max = 1, mean = 0.5,
# standard deviation = 0.3.
# Dla danych zeskalowanych z-scorem min = -2.43, max = 3.09, mean ~ 0, standard deviation = 1.
