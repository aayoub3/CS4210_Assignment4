#-------------------------------------------------------------------------
# AUTHOR: Abanob Ayoub
# FILENAME: deep_learning.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.

#importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28])) # input layer

    # iterate over the number of hidden layers to create the hidden layers
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu")) # hidden layers with ReLU activation function

    # output layer
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax")) # output layer with softmax activation function

    # defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# Using Keras to Load the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# For Fashion MNIST, the list of class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

n_hidden = [2, 5, 10]
n_neurons = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]

highest_accuracy = 0.0
best_model = None
best_parameters = {}

for h in n_hidden:
    for n in n_neurons:
        for l in l_rate:
            model = build_model(h, n, 10, l) # n_neurons_output is set to 10 for 10 classes

            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

            accuracy = model.evaluate(X_test, y_test)[1]

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_model = model
                best_parameters = {'Number of Hidden Layers': h, 'Number of Neurons': n, 'Learning Rate': l}

            print("Highest accuracy so far:", highest_accuracy)
            print("Parameters:", "Number of Hidden Layers:", h, ", Number of Neurons:", n, ", Learning Rate:", l)
            print()

# Printing the summary of the best model found
print(best_model.summary())
img_file = './model_arch.png'
tf.keras.utils.plot_model(best_model, to_file=img_file, show_shapes=True, show_layer_names=True)

# Plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()