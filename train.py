import os
import math

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

from augmentation import Augmentation

os.environ['CUDA_VISILE_DEVICES'] = '1'

def get_sequential_model(input_shape):
    model = keras.Sequential(
        [
            # Input
            layers.Input(input_shape), # size of input
            
            # 1st Conv Block
            layers.Conv2D(64, 3, strides = 1, activation = 'relu', padding = 'same'),
            layers.Conv2D(64, 3, strides = 1, activation = 'relu', padding = 'same'),
            layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # 2nd Conv Block
            layers.Conv2D(128, 3, strides = 1, activation = 'relu', padding = 'same'),
            layers.Conv2D(128, 3, strides = 1, activation = 'relu', padding = 'same'),
            layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Classifier
            layers.GlobalMaxPool2D(), # Bring 1 value in 1 filter
            layers.Dense(128, activation = 'relu'), # Fully Connected layer
            layers.Dense(1, activation = 'sigmoid')
        ]
    )

    return model

input_shape = [256, 256, 3]
model = get_sequential_model(input_shape)

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
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

            label = int(r['species']) - 1

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

class_name = ['Cat', 'Dog']
for batch in train_generator:
    X, y = batch
    plt.figure(figsize = (7, 7))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(X[i])
        plt.title(class_name[y[i]])
        plt.axis('off')

    break
# plt.show()

# Callback Functions
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss', patience = 3, verbose = 1,
    mode = 'min', restore_best_weights = False
) # If validation loss does not decrease while 3 epochs, it stops.

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.1, patience = 10, verbose = 1,
    mode = 'min', min_lr = 0.001
) # If validation loss does not decrease while 10 epochs, it fixes learning rate.
# (It will not executed due to early_stopping (3epochs))

filepath = '{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor = 'val_loss', verose = 1, save_best_only = True,
    save_weights_only = False, mode = 'min'
)

history = model.fit(
    train_generator,
    validation_data = valid_generator,
    epochs = 10,
    callbacks = [
        early_stopping,
        reduce_on_plateau,
        model_checkpoint
    ],
    verbose = 1 # logging
)

print(history.history)

import matplotlib.pyplot as plt
history = history.history

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label = 'train')
plt.plot(history['val_loss'], label = 'val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label = 'train')
plt.plot(history['val_accuracy'], label = 'val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.show()