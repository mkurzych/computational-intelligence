# baseline model for the dogs vs cats dataset
import os
import sys

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # Get the number of images in the training directory
    num_train_images = len(os.listdir('dataset_dogs_vs_cats/train/'))

    # Get the number of images in the test directory
    num_test_images = len(os.listdir('dataset_dogs_vs_cats/test/'))

    # Set the batch size
    batch_size = 64

    # Calculate the steps per epoch for the training and test sets
    train_steps = num_train_images // batch_size
    test_steps = num_test_images // batch_size

    # define model
    model = define_model()
    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                       width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                                 class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                               class_mode='binary', batch_size=64, target_size=(200, 200))
    checkpoint = ModelCheckpoint('model_cvd.keras', monitor='val_accuracy', save_best_only=True)
    # fit model
    history = History()
    model.fit(train_it, steps_per_epoch=train_steps,
              validation_data=test_it, validation_steps=test_steps, epochs=20, verbose=0,
              callbacks=[history, checkpoint])
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)

    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


# entry point, run the test harness
run_test_harness()
