#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

from tensorflow import keras
import tensorflow as tf
import numpy as np

#Sets the class naes
class_names = ["strawberry", "cherry", "tomato"]

#Loads keras model created in training.
def load_model():
    model = keras.models.load_model('saved_model/final_model')
    model.summary()
    return model

#Will run on the test data if the folder is not empty.
def test_data():
    data_dir = pathlib.Path('testdata/')

    images = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(100, 100),
    )

    return images


mean_confidence = [] #To see overall confidence in the model
#Pulls all the pieces together.
def run():
    model = load_model()
    images = test_data()
    pred = model.predict(images)

    for i in range(len(pred)):
        prediction = pred[i]
        score = tf.nn.softmax(prediction)
        mean_confidence.append(100 * np.max(score))
        print(
            "\nThis image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        actual = images.file_paths[i]
        print(
            "The real image is: {}".format(actual)
        )

if __name__ == "__main__":
    run()


import statistics
print(statistics.mean(mean_confidence))

