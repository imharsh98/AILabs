# Fashion MNIST classification using sequential API

# Using Keras to load the dataset

from tensoflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

# Create a validation set & standardization

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# Create a list of class names & plot image

import numpy as np
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(np.array(class_names)[0:10])

# Plot image

import matplotlib.pyplot as plt
some_cloth=X_train[0]
some_cloth_image=some_cloth.reshape(28,28)
plt.imshow(some_cloth_image,cmap="binary")
plt.axis("off")
plt.show()

# Creating the model using the Sequential API

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, qctivation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Display model information

model.summary()
model.layers
hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()

# Compiling the model

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])

# Training and evaluating the model

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Learning curves

import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Set the vertical range to [0-1]
plt.show()

# Model evaluation by test set

model.evaluate(X_test, y_test)

# Using the model to make predictions

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_test_proba=model.predict(X_test)
y_test_pred = np.argmax(y_test_proba, axis=1)
print(y_test_pred)
from sklearn.metrics import accuracy_score
print(np.array(class_names)[y_test_pred])

print("test set score: %f" % accuracy_score(y_test,y_test_pred))

