import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='tanh',
          input_shape=(X_train.shape[1],)),
    Dense(64, activation='tanh'),
    Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=16)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# a) Co robi StandardScaler? Jak transformowane są dane liczbowe?
# StandardScaler przekształca dane tak, aby miały średnią 0 i wariancję 1. Wzór na przekształcenie danych
# wygląda następująco: x' = (x - mean(x)) / std(x), gdzie x to dane, x' to przekształcone dane, mean(x) to
# średnia wartość danych, a std(x) to odchylenie standardowe danych.

# b) Czym jest OneHotEncoder (i kodowanie „one hot” ogólnie)? Jak etykiety klas są transformowane przez ten encoder?
# OneHotEncoder przekształca etykiety klas na wektory binarne. Każda klasa jest reprezentowana przez wektor
# binarny, w którym jedna z wartości jest równa 1, a pozostałe są równe 0. Na przykład, jeśli mamy 3 klasy
# (0, 1, 2), to OneHotEncoder przekształca je na wektory binarne (1, 0, 0), (0, 1, 0) i (0, 0, 1).


# c) Model ma 4 warstwy: wejściową, dwie ukryte warstwy z 64 neuronami każda i warstwę wyjściową.
# Ile neuronów ma warstwa wejściowa i co oznacza X_train.shape[1]? Ile neuronów ma warstwa wyjściowa
# i co oznacza y_encoded.shape[1]?
# Warstwa wejściowa ma tyle neuronów, ile jest cech w danych wejściowych. X_train.shape[1] oznacza liczbę
# cech w danych treningowych. Warstwa wyjściowa ma tyle neuronów, ile jest klas w danych wyjściowych.
# y_encoded.shape[1] oznacza liczbę klas w danych wyjściowych.

# d) Czy funkcja aktywacji relu jest najlepsza do tego zadania? Spróbuj użyć innej funkcji i obejrzyj wyniki
# relu: accuracy: 0.9665 - loss: 0.0745 - val_accuracy: 0.9524 - val_loss: 0.2438
# tanh: accuracy: 0.9528 - loss: 0.0858 - val_accuracy: 1.0000 - val_loss: 0.0646
# sigmoid: accuracy: 0.9645 - loss: 0.2386 - val_accuracy: 0.8095 - val_loss: 0.3775
# softmax: accuracy: 0.6576 - loss: 1.0502 - val_accuracy: 0.5238 - val_loss: 1.0588
# Najlepsze wyniki uzyskano dla funkcji aktywacji tanh. Funkcja aktywacji sigmoid i softmax nie nadaje się do
# tego zadania, ponieważ są przeznaczone do problemów klasyfikacji binarnej i wieloklasowej, odpowiednio.

# e) Model jest konfigurowany do treningu za pomocą polecenia compile. Tutaj wybieramy optymalizator (algorytm,
# który używa gradientu straty do aktualizacji wag), funkcję straty, metrykę do oceny modelu. Eksperymentuj ze
# zmianą tych parametrów na inne i uruchom program. Czy różne optymalizatory lub funkcje straty dają różne wyniki?
# Czy możemy dostosować szybkość uczenia się w optymalizatorze?
# funkcja straty: categorical_crossentropy, metryka: accuracy
# adam: accuracy: 0.9528 - loss: 0.0858 - val_accuracy: 1.0000 - val_loss: 0.0646
# sgd: accuracy: 0.9055 - loss: 0.2835 - val_accuracy: 0.9048 - val_loss: 0.3416
# rmsprop: accuracy: 0.9528 - loss: 0.0831 - val_accuracy: 1.0000 - val_loss: 0.0588
# adagrad: accuracy: 0.8757 - loss: 0.4501 - val_accuracy: 0.8571 - val_loss: 0.5111
# Najlepsze wyniki uzyskano dla optymalizatora adam. Optymalizatory sgd, rmsprop i adagrad dają gorsze wyniki.
# Tak, można dostosować szybkość uczenia się w optymalizatorze, ustawiając parametr learning_rate.

# f) W linii model.fit sieć neuronowa jest trenowana. Czy jest sposób, by zmodyfikować tę linię tak, aby rozmiar
# partii był równy 4 lub 8 lub 16? Jak wyglądają krzywe uczenia się dla różnych parametrów? Jak zmiana partii
# wpływa na kształt krzywych? Wypróbuj różne wartości i uruchom program.
# batch_size=4: accuracy: 0.9427 - loss: 0.1033 - val_accuracy: 1.0000 - val_loss: 0.0598
# batch_size=8: accuracy: 0.9692 - loss: 0.0758 - val_accuracy: 1.0000 - val_loss: 0.0549
# batch_size=16: accuracy: 0.9621 - loss: 0.0644 - val_accuracy: 1.0000 - val_loss: 0.0374
# Najlepsze wyniki uzyskano dla partii o rozmiarze 16. Zmiana rozmiaru partii wpływa na kształt krzywych uczenia
# się. Dla małych partii krzywe są bardziej nieregularne, a dla dużych partii są bardziej gładkie.

# g) Co możesz powiedzieć o wydajności sieci neuronowej na podstawie krzywych uczenia? W której epoce sieć
# osiągnęła najlepszą wydajność? Czy ta krzywa sugeruje dobrze dopasowany model, czy mamy do czynienia z
# niedouczeniem lub przeuczeniem?
# Sieć neuronowa osiągnęła najlepszą wydajność w 33 epoce. Krzywe uczenia sugerują dobrze dopasowany model,
# ponieważ krzywe treningowe i walidacyjne są blisko siebie i mają podobne wartości. Nie ma oznak niedouczenia
# ani przeuczenia.

# h) Przejrzyj niżej wymieniony kod i wyjaśnij co się w nim dzieje.
# W poniższym kodzie ładujemy dane iris i zapisany model. Następnie dotrenowujemy model na danych treningowych
# i testowych przez 10 epok. Na końcu zapisujemy nowy model i wyświetlamy dokładność modelu.
