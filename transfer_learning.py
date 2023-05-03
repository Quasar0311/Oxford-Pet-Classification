import os
import math
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.applications import EfficientNetB0
from augmentation import Augmentation
from train import DataGenerator

# Use pretrained model -> Transfer learning
# github.com/keras-team/keras-applications

# We use EfficientNet-B0

def get_model(input_shape):

    inputs = keras.Input(input_shape)
    base_model = EfficientNetB0(
        input_shape = input_shape,
        weights = 'imagenet',
        include_top = False,
        pooling = 'avg'
    )

    x = base_model(inputs)
    output = layers.Dense(1, activation = 'sigmoid')(x)
    model = keras.Model(inputs, output)

    return model

input_shape = (256, 256, 3)
model = get_model(input_shape)

adam = keras.optimizers.Adam(lr = 0.0001)

model.compile(
    optimizer = adam,
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
)

model.summary()

csv_path = 'data/kfolds.csv'
train_generator = DataGenerator(
    fold = 1,
    mode = 'train',
    csv_path = csv_path,
    batch_size = 128,
    image_size = 256,
    shuffle = True
)
valid_generator = DataGenerator(
    fold = 1,
    mode = 'val',
    csv_path = csv_path,
    batch_size = 128,
    image_size = 256,
    shuffle = True
)

history = model.fit(
    train_generator,
    validation_data = valid_generator,
    epochs = 10,
    verbose = 1
)