# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:51:48 2021

@author: priyanka
"""

from NeuralNetwork import *
from helper_functions import *

import tensorflow as tf
from matplotlib import pyplot


import numpy as np 
import math 

def main():
    

    #get data 
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()    

    #normalize features 
    X_train, X_test = normalizeX(X_train, X_test)    
    #OHE target 
    y_train, y_test = oheY(y_train, y_test)
    
    print("X train: ", X_train.shape, "Y train: ",  y_train.shape,  "X test: ", X_test.shape,  "Y_test: ", y_test.shape)
    
    num_features = 784 
    num_classes = 10
    num_neurons = [500, 300]
    activation_fxn = ['sigmoid', 'sigmoid']
    learning_rate = 0.1
    epochs = 10
    batch_size = -1
    
    myNN = NeuralNetwork(num_features, num_classes, num_neurons, activation_fxn, learning_rate, epochs, batch_size)
    myNN.train(X_train, y_train)
    t_accuracy = myNN.predict(X_test, y_test)
    print("Test: ", t_accuracy)

    
    
if __name__ == "__main__":
    main()