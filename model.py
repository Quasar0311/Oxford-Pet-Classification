# keras Sequential - 1 input & 1 output (Simple model)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

import os
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

# model.summary()


# keras Functional - multiple input & output
def get_functional_model(input_shape):
    inputs = keras.Input(input_shape)

    # 1st Conv Block
    x = layers.Conv2D(64, 3, strides = 1, activation = 'relu', padding = 'same')(inputs)
    x = layers.Conv2D(64, 3, strides = 1, activation = 'relu', padding = 'same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # 2nd Conv Block
    x = layers.Conv2D(128, 3, strides = 1, activation = 'relu', padding = 'same')(x)
    x = layers.Conv2D(128, 3, strides = 1, activation = 'relu', padding = 'same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Classifier
    x = layers.GlobalMaxPool2D()(x) # Bring 1 value in 1 filter
    x = layers.Dense(128, activation = 'relu')(x) # Fully Connected layer
    outputs = layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(inputs, outputs)

    return model

input_shape = [256, 256, 3]
model = get_functional_model(input_shape)

# model.summary()


# Model subclassing - Override methods
# Can customize several methods (ex. model.fit(), model.evaluate()...)

class SimpleCNN(keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_block1 = keras.Sequential(
            [
                layers.Conv2D(64, 3, strides = 1, activation = 'relu', padding = 'same'),
                layers.Conv2D(64, 3, strides = 1, activation = 'relu', padding = 'same'),
                layers.MaxPool2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
            ], name = 'conv_block_1'
        )
        self.conv_block2 = keras.Sequential(
            [
                layers.Conv2D(128, 3, strides = 1, activation = 'relu', padding = 'same'),
                layers.Conv2D(128, 3, strides = 1, activation = 'relu', padding = 'same'),
                layers.MaxPool2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
            ], name = 'conv_block_2'
        )
        self.classifier = keras.Sequential(
            [
                layers.GlobalMaxPool2D(), # Bring 1 value in 1 filter
                layers.Dense(128, activation = 'relu'), # Fully Connected layer
                layers.Dense(1, activation = 'sigmoid')
            ], name = 'classifier'
        )

    def call(self, input_tensor, training = False):
        x = self.conv_block1(input_tensor)
        x = self.conv_block2(x)
        x = self.classifier(x)

        return x

input_shape = [None, 256, 256, 3] # None -> Batch axis
model = SimpleCNN()
model.build(input_shape)

model.summary()

# Model Compiling - Optimizer, loss, metric selecting stage
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
)
