#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

#Set the parameters
batch_size = 50

#Images have been made smaller to help with processing.
img_height = 100
img_width = 100

#Calls in data from the training file. Splits into training (70%) and test sets (30%).
def get_data(batch_size, img_height, img_width):
    data_dir = pathlib.Path("traindata/")

    train_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=309,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=309,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    #Helps with processing by keeping a copy of data within the computer memory and overlapping processing.
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_data, test_data

#Creates a model using Keras TensorFlow.
def create_model(img_height, img_width):
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_height,img_width,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )
    
    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.5), #Sets the activiation function to 0, helps with overfitting
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(3, name="outputs")
    ])
    
    #Applies optimizer adam and loss function Sparse Categorical Cross Entropy.
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

#Trains the model with 25 epochs.
def train_model(train_data, test_data, model):
    epochs = 25
    history = model.fit(
      train_data,
      validation_data=test_data,
      epochs=epochs
    )
    test_loss, test_acc = model.evaluate(test_data, verbose=2)
    print("Test Accuracy: ", test_acc)

#Pulls all the pieces together
def create_train_model():
    train_data, test_data = get_data(batch_size, img_height, img_width)
    model = create_model(img_height, img_width)
    model.summary()
    train_model(train_data, test_data, model)

def main():
    create_train_model()

if __name__ == "__main__":
    main()

