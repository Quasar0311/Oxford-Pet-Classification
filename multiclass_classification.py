from email.mime import image
import os
import math

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.applications import EfficientNetB0
from augmentation import Augmentation

csv_path = 'data/kfolds.csv'

df = pd.read_csv(csv_path)
np.unique(df['id'])

value_counts = df['id'].value_counts().sort_index()
plt.figure(figsize = (10, 5))
plt.bar(range(len(value_counts)), value_counts.values)
plt.xticks(range(len(value_counts)), value_counts.index.values)
plt.tight_layout()
# plt.show()

def get_model(input_shape):

    inputs = keras.Input(input_shape)
    base_model = EfficientNetB0(
        input_shape = input_shape,
        weights = 'imagenet',
        include_top = False,
        pooling = 'avg'
    )

    x = base_model(inputs)
    output = layers.Dense(37, activation = 'softmax')(x)
    model = keras.Model(inputs, output)

    return model

input_shape = (256, 256, 3)
model = get_model(input_shape)

adam = keras.optimizers.Adam(lr = 0.0001)

model.compile(
    optimizer = adam,
    loss = 'sparse_categorical_crossentropy',
    metrics = 'accuracy'
)

model.summary()


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, csv_path,
                fold, image_size, mode = 'train', shuffle = True):
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.fold = fold
        self.mode = mode

        self.df = pd.read_csv(csv_path)

        if self.mode == 'train':
            self.df = self.df[self.df['fold'] != self.fold]
        elif self.mode == 'val':
            self.df = self.df[self.df['fold'] == self.fold]
        
        ### Remove invalid files (Error occurs)
        invalid_filenames = [
            'Egyptian_Mau_14',
            'Egyptian_Mau_139',
            'Egyptian_Mau_145',
            'Egyptian_Mau_156',
            'Egyptian_Mau_167',
            'Egyptian_Mau_177',
            'Egyptian_Mau_186',
            'Egyptian_Mau_191',
            'Ayssinian_5',
            'Ayssinian_34',
            'chihuahua_121',
            'beagle_116'
        ]
        self.df = self.df[-self.df['file_name'].isin(invalid_filenames)]

        self.transform = Augmentation(image_size, mode)

        self.on_epoch_end()

    def __len__(self):
        # Consider float case
        return math.ceil(len(self.df) / self.batch_size)
    
    def __getitem__(self, idx):
        strt = idx * self.batch_size
        fin = (idx + 1) * self.batch_size
        data = self.df.iloc[strt : fin]

        batch_x, batch_y = self.get_data(data)

        return np.array(batch_x), np.array(batch_y)

    def get_data(self, data):
        batch_x = []
        batch_y = []

        for _, r in data.iterrows():
            file_name = r['file_name']
            image = cv2.imread(f'data/images/{file_name}.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (self.image_size, self.image_size))

            if self.mode == 'train':
                image = image.astype('uint8')
                image = self.transform(image = image)

            # Rescaling
            image = image.astype('float32')
            image = image / 255.

            label = int(r['id']) - 1

            batch_x.append(image)
            batch_y.append(label)
        
        return batch_x, batch_y
    
    # Callback Function - Runs when each epoch ends
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac = 1).reset_index(drop = True)

csv_path = 'data/kfolds.csv'

train_generator = DataGenerator(
    fold = 1,
    mode = 'train',
    csv_path = csv_path,
    batch_size= 128,
    image_size= 256,
    shuffle= True
)

valid_generator = DataGenerator(
    fold = 1,
    mode = 'val',
    csv_path = csv_path,
    batch_size= 128,
    image_size= 256,
    shuffle= True
)

history = model.fit(
    train_generator,
    validation_data = valid_generator,
    epochs = 10,
    verbose = 1
)