import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

__dir__ = "/".join(__file__.rsplit('/')[:-1])

# Returns a base CNN model for MNIST
# UNCOMPILED
def get_mnist_cnn_uncompiled():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model

# Returns a base CNN model for MNIST with no training
def get_untrained_mnist_cnn():
    model = get_mnist_cnn_uncompiled()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

# Adds pre-trained weights to base CNN model for MNIST
# Compiles and returns pre-trained model
def get_pretrained_mnist_cnn():
    model = get_mnist_cnn_uncompiled()
    model.load_weights(__dir__ + '/data/mnist.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model
