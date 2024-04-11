# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, decomposition


iris = datasets.load_iris()
X = iris.data
y = iris.target

pca_iris = decomposition.PCA(n_components=3)
pca_iris.fit(X)
X = pca_iris.transform(X)


# Dokonaj PCA na bazie danych. Przyjrzyj się nowym kolumnom i wariancjom.
# Ile kolumn można usunąć, tak aby zachować minimum 95% wariancji (strata
# informacji nie może być większa niż 5%)? Korzystając z poniższego wzoru,
# swoją odpowiedź uzasadnij.


def information_loss(pca, i):
    return sum(pca.explained_variance_ratio_[-i:]) / sum(pca.explained_variance_ratio_[:])


print(pca_iris.explained_variance_ratio_)

print(information_loss(pca_iris, 1))
print(information_loss(pca_iris, 1) < 0.05)

print(information_loss(pca_iris, 2))
print(information_loss(pca_iris, 2) < 0.05)


# Bazę danych z usuniętymi kolumnami zobrazuj na wykresie punktowym, gdzie
# każdy punkt to irys. Jeśli w bazie zostawisz 2 kolumny, to wykres będzie
# na płaszczyźnie, a jeśli 3, to będzie trójwymiarowy.


plt.cla()

fig = plt.figure(1, figsize=(4, 3))
plt.clf()

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 0].mean(),
        X[y == label, 1].mean() + 1.5,
        X[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )

y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()
