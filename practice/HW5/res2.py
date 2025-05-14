import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", input_filters=None, **kwargs):
        super().__init__(**kwargs)
        bottleneck_filters = filters // 4
        self.activation_fn = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(bottleneck_filters, 1, strides=strides, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),

            tf.keras.layers.Conv2D(bottleneck_filters, 3, strides=1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1 or filters != kwargs.get("input_filters", filters):
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation_fn(Z + skip_Z)

class ResNet50(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.resize = tf.keras.layers.Resizing(224, 224, interpolation="bilinear", input_shape=input_shape)
        self.initial_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        ])
        self.residual_blocks = tf.keras.Sequential()
        self._build_resnet50_blocks()
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")
    
    def _build_resnet50_blocks(self):
        filters_per_stage = [256, 512, 1024, 2048]
        blocks_per_stage = [3, 4, 6, 3] # ResNet-50 architecture
        prev_filters = 64
        for filters, blocks in zip(filters_per_stage, blocks_per_stage):
            for block in range(blocks):
                strides = 1 if block > 0 else (1 if filters == prev_filters else 2)
                self.residual_blocks.add(BottleneckBlock(filters, strides=strides, input_filters=prev_filters))
                prev_filters = filters
    
    def call(self, inputs):
        x = self.resize(inputs)
        x = self.initial_layers(x)
        x = self.residual_blocks(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.classifier(x)

# Load Data