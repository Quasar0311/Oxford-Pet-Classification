# Increase data by giving some effects to data
# reverse left/right, color change, filter etc...

# Albumentations
# github.com/albumentations-team/albumentations

import os
import math
import random

import cv2
import albumentations as A

class Augmentation:
    def __init__(self, size, mode = 'train'):
        if mode == 'train':
            # Declare an augmentation pipeline
            self.transform = A.Compose([
                A.HorizontalFlip(p = 0.5),
                A.ShiftScaleRotate(
                    p = 0.5,
                    shift_limit = 0.05, # 5%
                    scale_limit = 0.05,
                    rotate_limit = 15,
                ),
                A.CoarseDropout(
                    p = 0.5,
                    max_holes = 8,
                    max_height = int(0.1 * size),
                    max_width = int(0.1 * size)
                ),
                A.RandomBrightnessContrast(p = 0.2)
            ])
    
    def __call__(self, **kwargs):
        if self.transform:
            augmented = self.transform(**kwargs)
            img = augmented['image']
            return img

