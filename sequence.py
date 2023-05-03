import math
import random
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras


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
            # Rescaling
            image = image / 255.

            label = int(r['species']) - 1

            batch_x.append(image)
            batch_y.append(label)
        
        return batch_x, batch_y
    
    # Callback Function - Runs when each epoch ends
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac = 1).reset_index(drop = True)


if __name__ == '__main__':
    csv_path = 'data/kfolds.csv'
    train_generator = DataGenerator(
        batch_size = 9,
        csv_path = csv_path,
        fold = 1,
        image_size = 256,
        mode = 'train',
        shuffle = True
    )

    print(len(train_generator))

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
    plt.show()