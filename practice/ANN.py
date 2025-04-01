# Fashion MNIST classification using sequential API

# Using Keras to load the dataset

from tensoflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
