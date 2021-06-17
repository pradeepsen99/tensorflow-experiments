'''
This is my interpretation from the google's version of Tensorflow documentation. 
Derived from https://www.tensorflow.org/tutorials/keras/classification
'''

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#Download the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

#Get testing and training datasets. 60k images used to rain the network and 10k used to verify it.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#This is the types of items present in the dataset. 10 types of 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Showing the sizes of dataset
print("Size of training dataset")
print(train_images.shape)
print("Size of testing dataset")
print(test_images.shape)

#Pre-processing the data. Dividing by 255 to scale the values down to the range of 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#Creating the model. This is setting up layers in the network.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),          #Transforms the picture from a 2D array to a 1D array.
    tf.keras.layers.Dense(128, activation='sigmoid'),       #Densely connected Neural Network. Changing the activation function can produce different results, can make the program faster or slower.
    tf.keras.layers.Dense(10, activation="softmax")         #Each node contains a score that shows wether the item belongs to one of the class_names
])

#Compiling model
model.compile(optimizer='adam',                                                         #ADAM is the best optimizer. Usually outperforms SGD and such
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     #Want to minimize this functino by steering the model into the right direction.
              metrics=['accuracy'])                                                     #Monitoring the training and testing steps. This one keeps track of accuracy which shows what fraction of the images that are correctly classified.

#Training the model
model.fit(train_images, train_labels, epochs=10)        #model.fit runs the model and starts training it. Epochs is the number of runs required, more is better but runs longer.

#Evaluating accuracy. Checking accuracy statistics. 
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)