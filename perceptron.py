#-------------------------------------------------------------------------
# AUTHOR: Abanob Ayoub
# FILENAME: perceptron.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None)
X_training = np.array(df.values)[:, :64]
y_training = np.array(df.values)[:, -1]

df = pd.read_csv('optdigits.tes', sep=',', header=None)
X_test = np.array(df.values)[:, :64]
y_test = np.array(df.values)[:, -1]

highest_perceptron_accuracy = 0.0
best_perceptron_params = {}

highest_mlp_accuracy = 0.0
best_mlp_params = {}

for learning_rate in n:
    for shuffle in r:
        clf_perceptron = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
        clf_perceptron.fit(X_training, y_training)

        perceptron_accuracy = clf_perceptron.score(X_test, y_test)
        if perceptron_accuracy > highest_perceptron_accuracy:
            highest_perceptron_accuracy = perceptron_accuracy
            best_perceptron_params = {'learning_rate': learning_rate, 'shuffle': shuffle}

        print(f"Highest Perceptron accuracy so far: {highest_perceptron_accuracy:.2f}, "
            f"Parameters: learning rate={learning_rate}, shuffle={shuffle}")

for learning_rate in n:
    for shuffle in r:
        for neurons in [(5,), (10,), (15,), (20,)]:
            clf_mlp = MLPClassifier(activation='logistic', learning_rate_init=learning_rate,
                                    hidden_layer_sizes=neurons, shuffle=shuffle, max_iter=1000)
            clf_mlp.fit(X_training, y_training)

            mlp_accuracy = clf_mlp.score(X_test, y_test)
            if mlp_accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy = mlp_accuracy
                best_mlp_params = {'learning_rate': learning_rate, 'shuffle': shuffle, 'neurons': neurons}

            print(f"Highest MLP accuracy so far: {highest_mlp_accuracy:.2f}, "
                f"Parameters: learning rate={learning_rate}, shuffle={shuffle}, neurons={neurons}")