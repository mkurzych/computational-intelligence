import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True)
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history, checkpoint])
# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()

# a) Co się dzieje w preprocessing? Do czego służy funkcja reshape, to_categorical i np.argmax?
# reshape - zmienia kształt macierzy, to_categorical - zamienia kolumnę z etykietami na macierz binarną,
# np.argmax - zwraca indeksy maksymalnych wartości w macierzy.

# b) Jak dane przepływają przez sieć i jak się w niej transformują? Co każda z warstw dostaje na
# wejście i co wyrzuca na wyjściu?
# W sieci neuronowej dane przepływają od warstwy wejściowej do warstwy wyjściowej. Każda warstwa dostaje na
# wejście wynik poprzedniej warstwy, a na wyjściu zwraca wynik, który jest przekazywany do kolejnej warstwy.
# Warstwa wejściowa dostaje dane wejściowe, a warstwa wyjściowa zwraca wynik końcowy. Warstwy ukryte przekształcają
# dane wejściowe, aby model mógł nauczyć się reprezentacji danych.

# c) Jakich błędów na macierzy błędów jest najwięcej. Które cyfry są często mylone z jakimi innymi?
# Model najczęściej myli cyfry 5 z 3, 9 z 4, i 2 z 7.

# d) Co możesz powiedzieć o krzywych uczenia się. Czy mamy przypadek przeuczenia lub niedouczenia się?
# Krzywe uczenia pokazują, że model uczy się dobrze, ponieważ zarówno dokładność treningu, jak i walidacji
# rosną wraz z liczbą epok. Nie ma przeuczenia ani niedouczenia, ponieważ krzywe uczenia są bliskie sobie.

# e) Jak zmodyfikować kod programu, aby model sieci był zapisywany do pliku h5 co epokę, pod warunkiem,
# że w tej epoce osiągnęliśmy lepszy wynik?
# Można dodać callback ModelCheckpoint, który zapisuje model do pliku keras (h5 wywołuje błąd), jeśli
# osiągnięto lepszy wynik na zbiorze walidacyjnym. Przykład:
# from tensorflow.keras.callbacks import ModelCheckpoint
#
# checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True)
# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history, checkpoint])
