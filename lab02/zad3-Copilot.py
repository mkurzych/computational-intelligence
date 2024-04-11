# can you generate 3 plots from the iris1.csv file in
# which one contains original data, second is min-max
# normalized, and third is z-score normalized? only use
# sepal.length and sepal.width

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the data
irises = pd.read_csv("iris1.csv")

# Define the scalers and their labels
scalers = [None, MinMaxScaler(), StandardScaler()]
scaler_labels = ['Original', 'Min-Max Normalized', 'Z-Score Normalized']

# Define the iris varieties
iris_varieties = {"Setosa": "Setosa", "Versicolor": "Versicolor", "Virginica": "Virginica"}

# Create a subplot for each version of the data
fig, axs = plt.subplots(len(scalers), figsize=(10, 15))

for i, (scaler, label) in enumerate(zip(scalers, scaler_labels)):
    # Extract the sepal.length and sepal.width data
    X = irises[["sepal.length", "sepal.width"]].values
    y = irises["variety"].values

    # Apply the scaler if one is defined
    if scaler:
        X = scaler.fit_transform(X)

    # Create a scatter plot for each iris variety
    for name, v_label in iris_varieties.items():
        axs[i].scatter(X[y == v_label, 0], X[y == v_label, 1], label=name)

    # Set the labels and title
    axs[i].set_xlabel("sepal.length")
    axs[i].set_ylabel("sepal.width")
    axs[i].set_title(label)
    axs[i].legend()

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()