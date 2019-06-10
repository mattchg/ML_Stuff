# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:49:28 2019

@author: Matthew


Uses a neural network to classify images of clothing
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
print(tf.__version__)


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Import Data from Fashion Library
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Preprocess 
train_images = train_images / 255.0
test_images = test_images / 255.0

#Take a peek
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #First Layer
    keras.layers.Dense(128, activation=tf.nn.relu),#Second Layer
    keras.layers.Dense(456, activation=tf.nn.relu),#3rd Layer
    keras.layers.Dense(10, activation=tf.nn.softmax)#OUtput Layer
])
    
"""    
The first layer in this network, tf.keras.layers.Flatten, transforms the format
of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
Think of this layer as unstacking rows of pixels in the image and lining them up.
This layer has no parameters to learn; it only reformats the data.    
    
After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
 These are densely-connected, or fully-connected, neural layers.
 The first Dense layer has 128 nodes (or neurons). The second (and last) layer is a 10-node softmax 
 layer—this returns an array of 10 probability scores that sum to 1.
 Each node contains a score that indicates the probability that
 the current image belongs to one of the 10 classes.
"""

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
Loss function: measures the correctness of the output,
 see https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
 This is for integer encoding, cat-cross is for one hot encoded data
 Optimizer: Network update function
 Metric: The following example uses accuracy, the fraction of the images that are correctly classified.
"""
 
model.fit(train_images, train_labels, epochs=5)

"""
Training step, we feed training set + labels for gradient descent
epochs:
"""


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', 100*test_acc)

predictions = model.predict(test_images)
predictions[0]

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  x = np.random.randint(0,9999)
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(x, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(x, predictions, test_labels)
plt.show()

    
#Predict
# Grab an image from the test dataset
img = test_images[0]
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

