# Fashion MNIST classification using CNN

## Using Keras to load the dataset

from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)


## Create a validation set & standarization

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0


## Create a list of class names & plot image

import numpy as np
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(np.array(class_names)[0:10])


## Plot image
import matplotlib.pyplot as plt
some_cloth=X_train[0]
some_cloth_image=some_cloth.reshape(28,28)
plt.imshow(some_cloth_image,cmap="binary")
plt.axis("off")
plt.show()


# Creating CNN using Sequential AP

model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])


# Display model information

model.summary()


## Compiling the model

model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",metrics=["accuracy"])


## Training and evaluating the model

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


## Learning curves

import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


## Model evaluation by test set

model.evaluate(X_test, y_test)


## Using the model to make predictions

y_test_proba=model.predict(X_test)
y_test_pred = np.argmax(y_test_proba, axis=1)
print(y_test_pred)
from sklearn.metrics import accuracy_score
print(np.array(class_names)[y_test_pred])

print("Test set score: %f" % accuracy_score(y_test,y_test_pred))



# ===============  Implementing a ResNet-34 CNN Using Keras ===============

## Create a ResidualUnit layer:
class ResidualUnit(keras.layers.Layer):
   def __init__(self, filters, strides=1, activation="relu", **kwargs):
      super().__init__(**kwargs)
      self.activation = keras.activations.get(activation)
      self.main_layers = [
          keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
          keras.layers.BatchNormalization(),
          self.activation,
          keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
          keras.layers.BatchNormalization()]
      self.skip_layers = []
      if strides > 1:
         self.skip_layers = [
             keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
             keras.layers.BatchNormalization()]

   def call(self, inputs):
      Z = inputs
      for layer in self.main_layers:
          Z = layer(Z)
      skip_Z = inputs
      for layer in self.skip_layers:
         skip_Z = layer(skip_Z)
      return self.activation(Z + skip_Z)



## Build the ResNet-34 using a Sequential model

model = keras.models.Sequential()
model.add(keras.layers.Resizing(224, 224, interpolation="bilinear",input_shape=[28,28,1]))
model.add(keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()



# ===============  Use pretrained model from Keras ===============

from sklearn.datasets import load_sample_image

## Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255

import numpy as np
images = np.array([china, flower])
print(images.shape)


## Get a pretrained model of ResNet-50

from tensorflow import keras

model = keras.applications.resnet50.ResNet50(weights="imagenet")

model.summary()


## data preprocessing images

import tensorflow as tf
images_resized = tf.image.resize(images, [224, 224])
print(images_resized.shape)

inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)


## Predict

Y_proba = model.predict(inputs)

top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)

for image_index in range(len(images)):
   print("Image #{}".format(image_index))
   for class_id, name, y_proba in top_K[image_index]:
     print(" {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
   print()



## Plot image

import matplotlib.pyplot as plt
plt.imshow(images[1, :, :, :]) # plot 1st image's 2nd feature map
plt.show()



# ===============  Fashion MNIST classification using Xception ===============

## Using Keras to load the dataset

from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


## Create a validation set & standarization

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:15000] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:15000]
X_test = X_test / 255.0


## Create a list of class names & plot image

import numpy as np
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]



## Use the Xception
model = keras.applications.Xception(weights=None, input_shape=(224, 224, 1), classes=10)

model.summary()


## Compiling the model

model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",metrics=["accuracy"])


## Resize images

import tensorflow as tf

### [batch, height, width, channels]

X_train=X_train[...,tf.newaxis]
X_valid=X_valid[...,tf.newaxis]
X_test=X_test[...,tf.newaxis]

X_train = tf.image.resize(X_train, [224, 224])
X_valid = tf.image.resize(X_valid, [224, 224])
X_test = tf.image.resize(X_test, [224, 224])


## Training and evaluating the model

history = model.fit(X_train, y_train, epochs=5,
validation_data=(X_valid, y_valid))


## Learning curves

import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


## Model evaluation by test set

model.evaluate(X_test, y_test)
